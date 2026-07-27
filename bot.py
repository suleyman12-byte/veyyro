"""
Telegram бот для оценки внешности
Использует MediaPipe Face Mesh для анализа 468 точек лица
"""

import os
import io
import random
import logging
import numpy as np
import cv2
import mediapipe as mp
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ─── Настройки ───────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Не задана переменная окружения BOT_TOKEN!")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Анализатор лица ─────────────────────────────────────────────────────────
class FaceAnalyzer:
    """Анализ лица по 468 точкам MediaPipe Face Mesh"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    # ── Ключевые точки ────────────────────────────────────────────────────
    # Центр лба (между бровями)
    L_EYE_INNER = 133
    R_EYE_INNER = 362
    L_EYE_OUTER = 33
    R_EYE_OUTER = 263
    L_EYE_TOP = 159
    L_EYE_BOT = 145
    R_EYE_TOP = 386
    R_EYE_BOT = 374
    NOSE_TIP = 1
    MOUTH_L = 61
    MOUTH_R = 291
    MOUTH_TOP = 13
    MOUTH_BOT = 14
    UP_LIP = 0
    LOW_LIP = 17
    CHIN = 152
    FOREHEAD = 10
    L_CHEEK = 234
    R_CHEEK = 454
    L_BROW = 70
    R_BROW = 300
    NOSE_BRIDGE = 6

    def analyze(self, image_bytes: bytes) -> dict:
        """Принимает байты изображения, возвращает dict с оценкой."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Не удалось прочитать изображение")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            raise ValueError("Лицо не обнаружено")

        h, w = img.shape[:2]
        face = results.multi_face_landmarks[0]
        lm = [(int(p.x * w), int(p.y * h)) for p in face.landmark]

        # ── Замеры ───────────────────────────────────────────────────────
        dist = lambda a, b: np.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1])

        face_w = dist(self.L_CHEEK, self.R_CHEEK)
        face_h = dist(self.FOREHEAD, self.CHIN)

        # ── 1. Симметрия ─────────────────────────────────────────────────
        def sym(a, b):
            return abs(lm[a][0] - lm[b][0])

        sym_eyes    = 1 - min(sym(self.L_EYE_INNER, self.NOSE_TIP),
                              sym(self.R_EYE_INNER, self.NOSE_TIP)) / (face_w / 2 + 1e-6)
        sym_brows   = 1 - min(sym(self.L_BROW, self.NOSE_TIP),
                              sym(self.R_BROW, self.NOSE_TIP)) / (face_w / 2 + 1e-6)
        sym_mouth   = 1 - min(sym(self.MOUTH_L, self.NOSE_TIP),
                              sym(self.MOUTH_R, self.NOSE_TIP)) / (face_w / 2 + 1e-6)
        mid_top = (lm[self.L_EYE_INNER][0] + lm[self.R_EYE_INNER][0]) / 2
        mid_bot = (lm[self.MOUTH_L][0] + lm[self.MOUTH_R][0]) / 2
        nose_x  = lm[self.NOSE_TIP][0]
        v_sym   = 1 - abs(mid_top - nose_x) / (face_w / 2 + 1e-6)
        v_sym2  = 1 - abs(mid_bot - nose_x) / (face_w / 2 + 1e-6)
        symmetry = np.clip(np.mean([sym_eyes, sym_brows, sym_mouth, v_sym, v_sym2]) * 100, 0, 100)

        # ── 2. Пропорции (третьи лица) ───────────────────────────────────
        nose_h = dist(self.NOSE_BRIDGE, self.NOSE_TIP)
        mouth_chin = dist(self.MOUTH_BOT, self.CHIN)
        third_top = nose_h / (face_h + 1e-6)
        third_mid = abs(self.MOUTH_BOT - self.NOSE_TIP) / (face_h + 1e-6) if False else \
                    dist(self.NOSE_TIP, self.MOUTH_BOT) / (face_h + 1e-6)
        third_bot = mouth_chin / (face_h + 1e-6)
        ideal_third = 1 / 3
        prop = 100 - np.mean([abs(third_top - ideal_third),
                              abs(third_mid - ideal_third),
                              abs(third_bot - ideal_third)]) * 300
        proportions = np.clip(prop, 0, 100)

        # ── 3. Золотое сечение ────────────────────────────────────────────
        GR = 1.618
        nose_w = dist(self.L_EYE_INNER, self.R_EYE_INNER)
        eye_w  = dist(self.L_EYE_OUTER, self.R_EYE_OUTER)
        lip_w  = dist(self.MOUTH_L, self.MOUTH_R)
        chin_h = dist(self.NOSE_TIP, self.CHIN)

        ratios = [
            nose_h / (chin_h + 1e-6),
            lip_w  / (nose_w + 1e-6),
            eye_w  / (face_w + 1e-6),
            nose_w / (lip_w + 1e-6),
        ]
        golden = 100 - np.mean([abs(r - GR) for r in ratios]) * 80
        golden = np.clip(golden, 0, 100)

        # ── 4. Глаза ─────────────────────────────────────────────────────
        eye_open_l = dist(self.L_EYE_TOP, self.L_EYE_BOT) / (dist(self.L_EYE_OUTER, self.L_EYE_INNER) + 1e-6)
        eye_open_r = dist(self.R_EYE_TOP, self.R_EYE_BOT) / (dist(self.R_EYE_OUTER, self.R_EYE_INNER) + 1e-6)
        eye_openness = np.clip((eye_open_l + eye_open_r) / 2 * 100, 0, 100)
        eye_ratio = eye_w / (face_w + 1e-6)
        eye_size  = np.clip(eye_ratio * 200, 0, 100)
        eyes = np.mean([eye_openness, eye_size])

        # ── 5. Губы ──────────────────────────────────────────────────────
        lip_h = dist(self.UP_LIP, self.LOW_LIP)
        lip_w_val = dist(self.MOUTH_L, self.MOUTH_R)
        lip_fullness = np.clip(lip_h / (lip_w_val + 1e-6) * 200, 0, 100)
        lip_sym = 1 - abs(lm[self.MOUTH_L][1] - lm[self.MOUTH_R][1]) / (face_h + 1e-6)
        lip_sym *= 100
        lips = np.clip(np.mean([lip_fullness, lip_sym]), 0, 100)

        # ── 6. Кожа ──────────────────────────────────────────────────────
        x1, y1 = max(lm[self.L_CHEEK][0] - 20, 0), max(lm[self.FOREHEAD][1] - 20, 0)
        x2, y2 = min(lm[self.R_CHEEK][0] + 20, w), min(lm[self.CHIN][1] + 20, h)
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            smoothness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 150, 100)
            hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            skin_val = np.mean(hsv[:, :, 2]) / 255 * 100
            skin = np.clip(np.mean([smoothness, skin_val]), 0, 100)
        else:
            skin = 50

        # ── 7. Общая оценка (1..10) ──────────────────────────────────────
        total = (symmetry   * 0.30 +
                 proportions * 0.25 +
                 golden      * 0.20 +
                 eyes        * 0.10 +
                 lips        * 0.10 +
                 skin        * 0.05)

        score = 1 + (total / 100) * 9          # 1..10
        score = np.clip(score + random.uniform(-0.15, 0.15), 1, 10)

        return {
            "score": round(float(score), 1),
            "face_count": len(results.multi_face_landmarks),
            "details": {
                "symmetry":     round(float(symmetry), 1),
                "proportions":  round(float(proportions), 1),
                "golden_ratio": round(float(golden), 1),
                "eyes":         round(float(eyes), 1),
                "lips":         round(float(lips), 1),
                "skin":         round(float(skin), 1),
            },
        }


# ─── Тексты оценок ───────────────────────────────────────────────────────────
COMMENTS = {
    "10": ["Шикарно! Абсолютная симметрия!", "Эталон красоты!", "Модельные стандарты!"],
    "9":  ["Потрясающе!", "Очень красивое лицо!", "Высочайшая оценка!"],
    "8":  ["Отлично!", "Очень привлекательно!", "Выделяетесь из толпы!"],
    "7":  ["Хорошо!", "Приятная внешность!", "Выше среднего!"],
    "6":  ["Нормально.", "Приятное лицо.", "Есть что улучшить, но в целом ок."],
    "5":  ["Средненько.", "Ничего особенного.", "Обычная внешность."],
    "4":  ["Ниже среднего.", "Есть над чем работать.", "Не самая выдающаяся внешность."],
    "3":  ["Неудачный ракурс?", "Попробуйте другое фото.", "Может, освещение не на вашей стороне?"],
    "2":  ["Сложный случай...", "Фото не передаёт вашу красоту!", "Попробуйте другое фото!"],
    "1":  ["Это точно вы на фото?", "Попробуйте фото при хорошем освещении.", "Бот сомневается в результатах."],
}

PROGRESS = ["🔍 Анализирую...", "📐 Измеряю пропорции...", "✨ Оцениваю...", "📊 Считаю баллы..."]


def score_comment(score: float) -> str:
    key = str(int(round(score)))
    return random.choice(COMMENTS.get(key, COMMENTS["5"]))


def bar(n: float, size: int = 10) -> str:
    filled = int(round(n / 100 * size))
    return "█" * filled + "░" * (size - filled)


# ─── Хэндлеры ────────────────────────────────────────────────────────────────
analyzer = FaceAnalyzer()


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот для оценки внешности!\n\n"
        "📸 Отправьте мне фото лица, и я:\n"
        "  • Найду лицо\n"
        "  • Проанализирую 468 точек\n"
        "  • Оценю по 6 критериям\n\n"
        "⚙️ Критерии оценки:\n"
        "  • Симметрия лица\n"
        "  • Пропорции (третьи лица)\n"
        "  • Золотое сечение\n"
        "  • Глаза\n"
        "  • Губы\n"
        "  • Кожа\n\n"
        "🎯 Оценка: 1-10 баллов\n\n"
        "💡 Советы для лучшего результата:\n"
        "  • Фронтальный ракурс\n"
        "  • Хорошее освещение\n"
        "  • Без очков и масок\n"
        "  • Одно лицо на фото\n\n"
        "Просто отправьте фото!"
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Как пользоваться:\n\n"
        "1. Отправьте фото лица\n"
        "2. Дождитесь анализа\n"
        "3. Получите оценку 1-10!\n\n"
        "Команды:\n"
        "/start - Начало\n"
        "/help  - Помощь\n\n"
        "💡 Фото должно содержать чёткое лицо в анфас."
    )


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message

    # Показываем "печатает..."
    await ctx.bot.send_chat_action(chat_id=msg.chat.id, action="typing")

    status = await msg.reply_text(random.choice(PROGRESS))

    try:
        photo = msg.photo[-1]
        file = await ctx.bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        image_bytes = buf.getvalue()

        result = analyzer.analyze(image_bytes)

        score = result["score"]
        d     = result["details"]
        emoji = score_comment(score)

        text = (
            f"🎯 Ваша оценка: {score}/10\n\n"
            f"💬 {emoji}\n\n"
            f"📊 Детали анализа:\n"
            f"  Симметрия:    {bar(d['symmetry'])}    {d['symmetry']}%\n"
            f"  Пропорции:    {bar(d['proportions'])}    {d['proportions']}%\n"
            f"  Золотое сечение: {bar(d['golden_ratio'])}    {d['golden_ratio']}%\n"
            f"  Глаза:        {bar(d['eyes'])}    {d['eyes']}%\n"
            f"  Губы:         {bar(d['lips'])}    {d['lips']}%\n"
            f"  Кожа:         {bar(d['skin'])}    {d['skin']}%\n\n"
            f"👤 Лиц на фото: {result['face_count']}\n\n"
            f"📸 Хотите попробовать другое фото?"
        )

        await status.edit_text(text)

    except ValueError as e:
        await status.edit_text(f"❌ {str(e)}\n\nПопробуйте другое фото!")

    except Exception as e:
        logger.error("Ошибка анализа: %s", e, exc_info=True)
        await status.edit_text(
            "❌ Произошла ошибка при анализе.\nПопробуйте другое фото!"
        )


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Отправьте фото для оценки!\n\n"
        "Для справки: /help"
    )


# ─── Запуск ──────────────────────────────────────────────────────────────────
def main():
    logger.info("Запуск бота...")
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Бот запущен! Ожидаю сообщения...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
