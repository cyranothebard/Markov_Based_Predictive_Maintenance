"""
Notebook Functionality Test

This script tests if the Jupyter notebooks can actually run without errors.
Tests for AI-generated code issues like missing imports, undefined variables, etc.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

project_root = Path(__file__).parent
notebooks_path = project_root / "notebooks"

def test_notebook_syntax(notebook_path):
    """Test if notebook has valid Python syntax in all code cells"""
    print(f"🧪 Testing syntax in {notebook_path.name}...")
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        print(f"   Found {len(code_cells)} code cells")
        
        syntax_errors = []
        for i, cell in enumerate(code_cells):
            cell_source = ''.join(cell['source'])
            if cell_source.strip():  # Skip empty cells
                try:
                    compile(cell_source, f'<cell {i}>', 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"Cell {i}: {e}")
        
        if syntax_errors:
            print(f"   ❌ {len(syntax_errors)} syntax errors found:")
            for error in syntax_errors[:3]:  # Show first 3 errors
                print(f"      {error}")
            return False
        else:
            print(f"   ✅ All code cells have valid syntax")
            return True
            
    except Exception as e:
        print(f"   ❌ Failed to parse notebook: {e}")
        return False

def check_notebook_imports(notebook_path):
    """Check if notebooks have obvious import issues"""
    print(f"🧪 Checking imports in {notebook_path.name}...")
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Extract all import statements
        imports = []
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        
        for cell in code_cells:
            cell_source = ''.join(cell['source'])
            lines = cell_source.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)
        
        print(f"   Found {len(imports)} import statements")
        
        # Check for common problematic imports
        problematic_imports = []
        for imp in imports:
            if 'src.' in imp and 'sys.path' not in ' '.join(imports):
                problematic_imports.append(f"Relative import without path setup: {imp}")
            if 'config' in imp.lower() and 'config/' not in str(notebook_path):
                problematic_imports.append(f"Config import may fail: {imp}")
        
        if problematic_imports:
            print("   ⚠️  Potential import issues found:")
            for issue in problematic_imports:
                print(f"      {issue}")
        else:
            print("   ✅ Imports look reasonable")
            
        return len(problematic_imports) == 0
        
    except Exception as e:
        print(f"   ❌ Failed to check imports: {e}")
        return False

def check_for_ai_slop_patterns(notebook_path):
    """Check for common AI-generated code patterns that may not work"""
    print(f"🧪 Checking for AI-generated issues in {notebook_path.name}...")
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        issues = []
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        
        for i, cell in enumerate(code_cells):
            cell_source = ''.join(cell['source'])
            
            # Common AI-generated issues
            if 'TODO:' in cell_source or 'FIXME:' in cell_source:
                issues.append(f"Cell {i}: Contains TODO/FIXME comments")
            
            if 'your_' in cell_source.lower() or 'path/to/' in cell_source:
                issues.append(f"Cell {i}: Contains placeholder paths/variables")
            
            if 'config.yaml' in cell_source and not os.path.exists(project_root / 'config.yaml'):
                issues.append(f"Cell {i}: References non-existent config.yaml")
                
            if 'except:' in cell_source:
                issues.append(f"Cell {i}: Uses bare except clause")
            
            # Check for hardcoded paths that may not exist
            if '/Users/' in cell_source or 'C:\\' in cell_source:
                issues.append(f"Cell {i}: Contains hardcoded absolute paths")
        
        if issues:
            print(f"   ⚠️  {len(issues)} potential AI-generated issues found:")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"      {issue}")
        else:
            print("   ✅ No obvious AI-generated issues detected")
            
        return len(issues) == 0
        
    except Exception as e:
        print(f"   ❌ Failed to check for issues: {e}")
        return False

def test_all_notebooks():
    """Test all notebooks in the notebooks directory"""
    print("🚀 Testing Jupyter notebooks functionality...")
    print("=" * 60)
    
    if not notebooks_path.exists():
        print("❌ Notebooks directory not found!")
        return
    
    notebook_files = list(notebooks_path.glob("*.ipynb"))
    
    if not notebook_files:
        print("❌ No notebook files found!")
        return
    
    print(f"Found {len(notebook_files)} notebooks to test:")
    for nb in notebook_files:
        print(f"  📓 {nb.name}")
    
    print("\n" + "=" * 60)
    
    results = {}
    for notebook_path in notebook_files:
        print(f"\n📓 Testing: {notebook_path.name}")
        print("-" * 40)
        
        syntax_ok = test_notebook_syntax(notebook_path)
        imports_ok = check_notebook_imports(notebook_path)
        no_ai_slop = check_for_ai_slop_patterns(notebook_path)
        
        results[notebook_path.name] = {
            'syntax_ok': syntax_ok,
            'imports_ok': imports_ok,
            'no_ai_slop': no_ai_slop,
            'overall': syntax_ok and imports_ok and no_ai_slop
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 NOTEBOOKS REVIEW SUMMARY:")
    print("=" * 60)
    
    for notebook, result in results.items():
        status = "✅ PASS" if result['overall'] else "❌ ISSUES"
        print(f"{status} {notebook}")
        if not result['overall']:
            issues = []
            if not result['syntax_ok']:
                issues.append("Syntax errors")
            if not result['imports_ok']:
                issues.append("Import issues")
            if not result['no_ai_slop']:
                issues.append("AI-generated issues")
            print(f"      Issues: {', '.join(issues)}")
    
    overall_pass = all(r['overall'] for r in results.values())
    if overall_pass:
        print("\n🎉 All notebooks appear functional!")
    else:
        print("\n🚨 Some notebooks have issues that need attention.")

if __name__ == "__main__":
    test_all_notebooks()