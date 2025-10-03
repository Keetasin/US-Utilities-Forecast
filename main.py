# main.py
from src.web import create_app, db
from src.web.scheduler import (
    start_stock_scheduler, start_news_scheduler, start_forecast_scheduler
)

app = create_app()

with app.app_context():
    # ต้อง import models ก่อนค่อย create_all()
    from src.web import models  # noqa: F401  (แค่ให้ตารางถูกโหลดเข้ามา)
    db.create_all()             # ✅ สร้างตารางใน Postgres ถ้ายังไม่มี

    # ถ้า DB ว่าง scheduler จะเติมข้อมูลรอบแรกให้เองตาม logic เดิม
    start_stock_scheduler(app)
    start_news_scheduler(app)
    start_forecast_scheduler(app)

if __name__ == "__main__":
    # ปิด debug/reloader เพื่อกัน APScheduler โดนสตาร์ทซ้ำ
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
