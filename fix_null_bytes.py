#!/usr/bin/env python
"""
Script to remove null bytes from Python files
"""
import os
import sys

def fix_file(filepath):
    """Remove null bytes from a file."""
    print(f"Fixing {filepath}...")
    
    try:
        # Read content with null bytes
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Remove null bytes
        clean_content = content.replace(b'\x00', b'')
        
        # Write clean content back
        if content != clean_content:
            with open(filepath, 'wb') as f:
                f.write(clean_content)
            print(f"  Fixed: Removed null bytes from {filepath}")
        else:
            print(f"  No null bytes found in {filepath}")
            
        return True
    except Exception as e:
        print(f"  Error fixing {filepath}: {e}")
        return False

def scan_directory(directory):
    """Recursively scan directory and fix Python files."""
    fixed_files = 0
    error_files = 0
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                result = fix_file(filepath)
                if result:
                    fixed_files += 1
                else:
                    error_files += 1
    
    return fixed_files, error_files

def main():
    """Main function."""
    directory = '.'  # Current directory
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    print(f"Scanning {directory} for Python files with null bytes...")
    fixed_files, error_files = scan_directory(directory)
    
    print(f"\nProcess completed.")
    print(f"Files processed: {fixed_files}")
    print(f"Files with errors: {error_files}")

if __name__ == "__main__":
    main() 