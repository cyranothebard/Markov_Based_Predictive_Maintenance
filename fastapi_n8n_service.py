"""
FastAPI Service for AI Process Automation Integration

This service provides REST endpoints for n8n workflows to consume 
predictive maintenance health state predictions and trigger automation.
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import uvicorn
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from data.data_loader import CMAPSSLoader
from data.feature_engineer import FeatureEngineer
from models.markov_model import MarkovChainRUL

app = FastAPI(
    title="AI Predictive Maintenance API",
    description="REST API for n8n automation workflows - Predictive maintenance health state monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for n8n integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your n8n instance URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class HealthStateResponse(BaseModel):
    equipment_id: str
    health_state: int
    health_state_name: str
    confidence: float
    predicted_rul: float
    urgency_level: str
    timestamp: str

class AutomationAlert(BaseModel):
    equipment_id: str
    alert_type: str
    priority: str
    primary_anomaly: str
    secondary_indicators: List[str]
    recommended_actions: List[str]
    predicted_rul: float
    confidence: float
    timestamp: str

class ProcessAutomationResponse(BaseModel):
    trigger_automation: bool
    alert_level: str
    equipment_count: int
    alerts: List[AutomationAlert]
    summary: Dict[str, Any]

# Global cache for health states (in production, use Redis)
health_cache = {}
last_analysis_time = None

def load_health_analysis():
    """Load the latest health analysis results"""
    results_file = project_root / 'results' / 'health_state_analysis.json'
    
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Health analysis not found. Run analyze_all_datasets.py first.")
    
    with open(results_file, 'r') as f:
        return json.load(f)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Predictive Maintenance API for n8n Process Automation",
        "version": "1.0.0",
        "endpoints": {
            "/health": "System health check",
            "/equipment/health": "Get equipment health states", 
            "/automation/alerts": "Get automation alerts for n8n",
            "/automation/trigger": "Check if automation should be triggered",
            "/equipment/{equipment_id}": "Get specific equipment details",
            "/docs": "Interactive API documentation"
        },
        "integration": "Designed for n8n workflow automation",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for n8n monitoring"""
    return {
        "status": "healthy",
        "service": "AI Predictive Maintenance API",
        "timestamp": datetime.now().isoformat(),
        "uptime": "System operational"
    }

@app.get("/equipment/health", response_model=List[HealthStateResponse])
async def get_equipment_health(
    health_state: Optional[int] = Query(None, description="Filter by health state (0=Healthy, 1=Warning, 2=Critical, 3=Failure)"),
    dataset: Optional[str] = Query(None, description="Filter by dataset (FD001, FD002, FD003, FD004)"),
    limit: int = Query(100, description="Maximum number of results")
):
    """
    Get equipment health states for n8n monitoring
    
    ### Usage for n8n:
    - **GET** `/equipment/health?health_state=2` - Get only critical equipment
    - **GET** `/equipment/health?health_state=1&limit=50` - Get warning equipment (limit 50)
    - **GET** `/equipment/health?dataset=FD001` - Get equipment from specific dataset
    """
    try:
        analysis_data = load_health_analysis()
        predictions = analysis_data['full_predictions']
        
        # Filter based on parameters
        filtered_predictions = predictions
        
        if health_state is not None:
            filtered_predictions = [p for p in filtered_predictions if p['health_state'] == health_state]
        
        if dataset:
            filtered_predictions = [p for p in filtered_predictions if p['dataset'] == dataset]
        
        # Limit results
        filtered_predictions = filtered_predictions[:limit]
        
        # Convert to response model
        responses = [
            HealthStateResponse(
                equipment_id=p['equipment_id'],
                health_state=p['health_state'],
                health_state_name=p['health_state_name'],
                confidence=p['confidence'],
                predicted_rul=p['predicted_rul'],
                urgency_level=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][p['health_state']],
                timestamp=p['timestamp']
            )
            for p in filtered_predictions
        ]
        
        return responses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving equipment health: {str(e)}")

@app.get("/automation/alerts", response_model=List[AutomationAlert])
async def get_automation_alerts(
    priority: Optional[str] = Query(None, description="Filter by priority (LOW, MEDIUM, HIGH, CRITICAL)"),
    limit: int = Query(50, description="Maximum number of alerts")
):
    """
    Get automation alerts for n8n workflow triggers
    
    ### n8n Integration Example:
    ```json
    {
        "equipment_id": "FD001_ENGINE_012",
        "alert_type": "CRITICAL_MAINTENANCE",
        "priority": "HIGH",
        "primary_anomaly": "Temperature sensor 14 exceeded safety threshold",
        "recommended_actions": ["Schedule immediate maintenance within 48 hours"]
    }
    ```
    """
    try:
        analysis_data = load_health_analysis()
        n8n_payloads = analysis_data['n8n_payloads']
        
        # Filter by priority if specified
        if priority:
            n8n_payloads = [p for p in n8n_payloads if p['urgency_level'] == priority]
        
        # Limit results
        n8n_payloads = n8n_payloads[:limit]
        
        # Convert to AutomationAlert model
        alerts = [
            AutomationAlert(
                equipment_id=alert['equipment_id'],
                alert_type=f"{alert['urgency_level']}_MAINTENANCE",
                priority=alert['urgency_level'],
                primary_anomaly=alert['analysis']['primary_anomaly'],
                secondary_indicators=alert['analysis']['secondary_indicators'],
                recommended_actions=alert['analysis']['recommended_actions'],
                predicted_rul=alert['predicted_rul'],
                confidence=alert['confidence'],
                timestamp=alert['timestamp']
            )
            for alert in n8n_payloads
        ]
        
        return alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving automation alerts: {str(e)}")

@app.get("/automation/trigger", response_model=ProcessAutomationResponse)
async def check_automation_trigger():
    """
    Check if process automation should be triggered
    
    ### n8n Workflow Decision Point:
    This endpoint returns a boolean `trigger_automation` that n8n can use 
    to decide whether to execute maintenance workflows.
    """
    try:
        analysis_data = load_health_analysis()
        automation_triggers = analysis_data['automation_triggers']
        
        # Determine if automation should trigger
        critical_count = len(automation_triggers['critical'])
        failure_count = len(automation_triggers['failure'])
        warning_count = len(automation_triggers['warning'])
        
        # Trigger logic: any critical or failure alerts
        should_trigger = critical_count > 0 or failure_count > 0
        
        # Determine alert level
        if failure_count > 0:
            alert_level = "CRITICAL"
        elif critical_count > 0:
            alert_level = "HIGH"
        elif warning_count > 0:
            alert_level = "MEDIUM"
        else:
            alert_level = "LOW"
        
        # Create alert objects for immediate action
        immediate_alerts = []
        
        # Add failure alerts (highest priority)
        for equipment in automation_triggers['failure']:
            immediate_alerts.append(AutomationAlert(
                equipment_id=equipment['equipment_id'],
                alert_type="EMERGENCY_MAINTENANCE",
                priority="CRITICAL",
                primary_anomaly=equipment['analysis']['primary_anomaly'],
                secondary_indicators=equipment['analysis']['secondary_indicators'],
                recommended_actions=equipment['analysis']['recommended_actions'],
                predicted_rul=equipment['predicted_rul'],
                confidence=equipment['confidence'],
                timestamp=equipment['timestamp']
            ))
        
        # Add critical alerts
        for equipment in automation_triggers['critical']:
            immediate_alerts.append(AutomationAlert(
                equipment_id=equipment['equipment_id'],
                alert_type="URGENT_MAINTENANCE",
                priority="HIGH",
                primary_anomaly=equipment['analysis']['primary_anomaly'],
                secondary_indicators=equipment['analysis']['secondary_indicators'],
                recommended_actions=equipment['analysis']['recommended_actions'],
                predicted_rul=equipment['predicted_rul'],
                confidence=equipment['confidence'],
                timestamp=equipment['timestamp']
            ))
        
        return ProcessAutomationResponse(
            trigger_automation=should_trigger,
            alert_level=alert_level,
            equipment_count=len(immediate_alerts),
            alerts=immediate_alerts[:10],  # Limit to top 10 for performance
            summary={
                "total_equipment": analysis_data['total_engines'],
                "equipment_needing_attention": analysis_data['degraded_equipment'],
                "warning_alerts": warning_count,
                "critical_alerts": critical_count,
                "failure_alerts": failure_count,
                "analysis_timestamp": analysis_data['analysis_timestamp'],
                "recommendation": "Immediate action required" if should_trigger else "Continue monitoring"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking automation trigger: {str(e)}")

@app.get("/equipment/{equipment_id}")
async def get_equipment_details(equipment_id: str):
    """
    Get detailed information about specific equipment
    
    ### Example:
    **GET** `/equipment/FD001_ENGINE_012`
    """
    try:
        analysis_data = load_health_analysis()
        
        # Find equipment in full predictions
        equipment = None
        for prediction in analysis_data['full_predictions']:
            if prediction['equipment_id'] == equipment_id:
                equipment = prediction
                break
        
        if not equipment:
            raise HTTPException(status_code=404, detail=f"Equipment {equipment_id} not found")
        
        # Check if it's in n8n payloads for analysis data
        analysis = None
        for payload in analysis_data['n8n_payloads']:
            if payload['equipment_id'] == equipment_id:
                analysis = payload['analysis']
                break
        
        response = {
            "equipment_id": equipment['equipment_id'],
            "dataset": equipment['dataset'],
            "unit": equipment['unit'],
            "current_cycle": equipment['current_cycle'],
            "health_state": equipment['health_state'],
            "health_state_name": equipment['health_state_name'],
            "confidence": equipment['confidence'],
            "predicted_rul": equipment['predicted_rul'],
            "actual_rul": equipment['actual_rul'],
            "state_probabilities": equipment['state_probabilities'],
            "timestamp": equipment['timestamp']
        }
        
        if analysis:
            response['analysis'] = analysis
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving equipment details: {str(e)}")

@app.get("/n8n/webhook/health-check")
async def n8n_webhook_health_check():
    """
    Specific endpoint for n8n webhook health monitoring
    Returns simplified format for n8n workflow consumption
    """
    try:
        analysis_data = load_health_analysis()
        automation_triggers = analysis_data['automation_triggers']
        
        return {
            "status": "active",
            "critical_equipment": len(automation_triggers['critical']),
            "failure_equipment": len(automation_triggers['failure']),
            "warning_equipment": len(automation_triggers['warning']),
            "trigger_automation": len(automation_triggers['critical']) > 0 or len(automation_triggers['failure']) > 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🚀 Starting AI Predictive Maintenance API")
    print("📡 Designed for n8n Process Automation Integration")
    print("=" * 60)
    print("Available endpoints:")
    print("• http://localhost:8000/docs - Interactive API documentation") 
    print("• http://localhost:8000/automation/trigger - Main n8n trigger endpoint")
    print("• http://localhost:8000/automation/alerts - Get alerts for workflows")
    print("• http://localhost:8000/equipment/health - Equipment health monitoring")
    print("• http://localhost:8000/n8n/webhook/health-check - n8n webhook format")
    print("=" * 60)
    
    # Check if analysis data exists
    results_file = project_root / 'results' / 'health_state_analysis.json'
    if not results_file.exists():
        print("⚠️  WARNING: Run 'python analyze_all_datasets.py' first to generate analysis data")
    else:
        print("✅ Health analysis data found - API ready for n8n integration")
    
    uvicorn.run(
        "fastapi_n8n_service:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )