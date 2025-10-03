import os

def save_project_structure(root_dir, output_file, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ["venv", "__pycache__", ".git", "notebooks", "logs", "plugins"]

    with open(output_file, "w", encoding="utf-8") as f:
        for current_dir, subdirs, files in os.walk(root_dir):
            # ตัดโฟลเดอร์ที่ไม่ต้องการออก
            subdirs[:] = [d for d in subdirs if d not in exclude_dirs]

            # ระดับความลึก
            level = current_dir.replace(root_dir, "").count(os.sep)
            indent = "│   " * (level - 1) + ("├── " if level > 0 else "")
            f.write(f"{indent}{os.path.basename(current_dir)}/\n")

            # แสดงไฟล์
            for i, filename in enumerate(files):
                prefix = "│   " * level
                connector = "└── " if i == len(files) - 1 else "├── "
                f.write(f"{prefix}{connector}{filename}\n")

# ใช้งาน
project_path = r"C:\Users\student\happy-stock\US-Utilities-Forecast"
output_file = "project_structure.txt"
save_project_structure(project_path, output_file)

print(f"✅ โครงสร้างถูกบันทึกที่ {output_file}")
