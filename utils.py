import os
from typing import Optional

def format_python_files_to_markdown(root_dir: str, output_filename: str = "code.md"):
    """
    Recursively searches a directory for .py files, reads their content,
    and writes the file path and content in Markdown format to the specified file.

    Args:
        root_dir: The path to the starting folder.
        output_filename: The name of the file to write the Markdown output to.
    """
    if not os.path.isdir(root_dir):
        print(f"Error: Directory not found at '{root_dir}'")
        return

    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.endswith('.py'):
                        file_path = os.path.join(dirpath, filename)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            relative_path = os.path.relpath(file_path, root_dir)
                            
                            markdown_output = (
                                f"\n## {relative_path}\n"
                                "```python\n"
                                f"{content.strip()}\n"
                                "```\n"
                            )
                            
                            outfile.write(markdown_output)
                            print(f"Successfully processed: {relative_path}")

                        except Exception as e:
                            outfile.write(f"\n## Error Reading File: {relative_path}\n")
                            outfile.write(f"```text\nError: {e}\n```\n")
                            print(f"Error reading file {relative_path}: {e}")
                            
        print(f"\n*** Success! All code has been written to: {os.path.abspath(output_filename)} ***")

    except IOError as e:
        print(f"Failed to write to output file {output_filename}: {e}")
        
if __name__ == "__main__":
    target_folder, output_file = "src", "code.md"
    format_python_files_to_markdown(target_folder, output_file)