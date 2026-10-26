from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COLORS = {
    "bg": "#f8fafc", "ink": "#0f172a", "muted": "#64748b", "line": "#cbd5e1",
    "blue": "#1d4ed8", "soft": "#eff6ff", "border": "#3b82f6", "white": "#ffffff",
}


def font(size, latin=False):
    return ImageFont.truetype(LATIN_FONT if latin else FONT, size)


def mixed_text(draw, xy, text, size=None, fill=None, **kwargs):
    if size is None:
        size = kwargs["font"].size
    x, y = xy
    for character in text:
        latin = ord(character) < 128 or character in "→←↑↓·–—×＋"
        draw.text((x, y), character, font=font(size, latin), fill=fill)
        x += draw.textlength(character, font=font(size, latin))


def center(draw, xy, text, size, fill, weight=False):
    x, y = xy
    width = sum(draw.textlength(character, font=font(size, ord(character) < 128 or character in "→←↑↓·–—×＋")) for character in text)
    mixed_text(draw, (x - width / 2, y), text, size, fill)


def box(draw, rect, title, lines, highlighted=False):
    draw.rounded_rectangle(rect, radius=16, fill=COLORS["soft"] if highlighted else COLORS["white"], outline=COLORS["border"] if highlighted else COLORS["line"], width=2 if highlighted else 1)
    x1, y1, x2, y2 = rect
    x = (x1 + x2) / 2
    center(draw, (x, y1 + 25), title, 21, COLORS["blue"] if highlighted else COLORS["ink"])
    for index, line in enumerate(lines):
        center(draw, (x, y1 + 59 + index * 26), line, 16, COLORS["muted"])


def arrow(draw, start, end, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        dx, dy = x2 - x1, y2 - y1
        steps = max(1, int(max(abs(dx), abs(dy)) / 10))
        for index in range(0, steps, 2):
            begin = index / steps
            finish = min((index + 1) / steps, 1)
            draw.line((x1 + dx * begin, y1 + dy * begin, x1 + dx * finish, y1 + dy * finish), fill=COLORS["muted"], width=2)
    else:
        draw.line((x1, y1, x2, y2), fill=COLORS["muted"], width=2)
    if y2 == y1 and x2 > x1:
        draw.polygon([(x2, y2), (x2 - 9, y2 - 4), (x2 - 9, y2 + 4)], fill=COLORS["muted"])
    elif y2 == y1 and x2 < x1:
        draw.polygon([(x2, y2), (x2 + 9, y2 - 4), (x2 + 9, y2 + 4)], fill=COLORS["muted"])
    elif y2 > y1:
        draw.polygon([(x2, y2), (x2 - 4, y2 - 9), (x2 + 4, y2 - 9)], fill=COLORS["muted"])
    elif y2 < y1:
        draw.polygon([(x2, y2), (x2 - 4, y2 + 9), (x2 + 4, y2 + 9)], fill=COLORS["muted"])


def base(width, height, title, subtitle):
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    mixed_text(draw, (58, 37), title, font=font(29), fill=COLORS["ink"])
    mixed_text(draw, (58, 77), subtitle, font=font(17), fill=COLORS["muted"])
    return image, draw


def render_method():
    image, draw = base(1500, 570, "VLA 交互式后训练流程", "稀疏二值奖励 · 单卡 LIBERO-Long 实验")
    boxes = [
        ((60, 170, 250, 305), "SFT 初始权重", ["官方 QueST", "策略初始化"]),
        ((285, 170, 490, 305), "并行轨迹采样", ["每个状态 K=16", "成功与失败轨迹"]),
        ((525, 170, 740, 305), "组内优势估计", ["留一法", "无需价值网络"]),
        ((775, 150, 1045, 325), "有界动态采样", ["过滤零优势轨迹组", "限制重试并记录终止", "防止无效策略更新"]),
        ((1080, 170, 1245, 305), "PPO 更新", ["15 轮训练", "保存最终权重"]),
        ((1280, 170, 1445, 305), "独立评估", ["50 个固定状态", "成功率与成本"]),
    ]
    for index, (rect, title, lines) in enumerate(boxes):
        box(draw, rect, title, lines, index == 3)
    for start, end in [((250, 237), (278, 237)), ((490, 237), (518, 237)), ((740, 237), (768, 237)), ((1045, 237), (1073, 237)), ((1245, 237), (1273, 237))]:
        arrow(draw, start, end)
    draw.line((1362, 315, 1362, 390, 387, 390, 387, 313), fill=COLORS["muted"], width=2)
    arrow(draw, (387, 390), (387, 305), dashed=True)
    center(draw, (875, 401), "交互 → 评价 → 更新 → 再次交互", 16, COLORS["muted"])
    draw.rounded_rectangle((60, 475, 80, 495), radius=4, fill=COLORS["soft"], outline=COLORS["border"])
    mixed_text(draw, (92, 474), "本项目：有界重试、诊断日志与受控评估", font=font(16), fill="#334155")
    draw.rounded_rectangle((795, 475, 815, 495), radius=4, fill=COLORS["white"], outline=COLORS["line"])
    mixed_text(draw, (827, 474), "官方 RIPT-VLA 基础实现", font=font(16), fill="#334155")
    mixed_text(draw, (60, 529), "范围：E2/E3 结果对应单个 LIBERO-Long 任务和一个训练 seed。", font=font(14), fill=COLORS["muted"])
    image.save(ROOT / "assets/method_pipeline.png")


def render_evolution():
    image, draw = base(1500, 610, "自进化 VLA：后训练、规划与验证回流", "价值学习冻结 VLA 主干 · 仿真验证结果 · 仅一轮经验回流")
    top = [
        ((60, 145, 250, 250), "SFT 策略", ["QueST 初始化"]),
        ((295, 145, 520, 250), "RIPT 后训练", ["RLOO · 有界采样"]),
        ((565, 145, 805, 250), "交互数据", ["成功与失败动作块"]),
        ((850, 145, 1100, 250), "结果价值模型", ["轻量 Q 代理"]),
        ((1145, 145, 1440, 250), "候选动作规划", ["评估 K 个动作块"]),
    ]
    for index, (rect, title, lines) in enumerate(top):
        box(draw, rect, title, lines, index >= 3)
    for start, end in [((250, 197), (288, 197)), ((520, 197), (558, 197)), ((805, 197), (843, 197)), ((1100, 197), (1138, 197))]:
        arrow(draw, start, end)
    bottom = [
        ((1145, 330, 1440, 435), "执行与验证", ["真实 LIBERO 结果"]),
        ((700, 330, 1000, 435), "受控评估", ["K=1/4/8 · 留出状态"]),
        ((255, 330, 555, 435), "一轮经验回流", ["仅验证成功轨迹"]),
    ]
    for rect, title, lines in bottom:
        box(draw, rect, title, lines)
    arrow(draw, (1292, 250), (1292, 323))
    arrow(draw, (1138, 382), (1007, 382))
    arrow(draw, (693, 382), (562, 382))
    arrow(draw, (405, 330), (405, 250), dashed=True)
    draw.rounded_rectangle((60, 500, 80, 520), radius=4, fill=COLORS["soft"], outline=COLORS["border"])
    mixed_text(draw, (92, 499), "本项目新增：结果价值排序与有界自训练回流", font=font(16), fill="#334155")
    mixed_text(draw, (60, 554), "同时保留正负结果。结果价值排序不是精确 Bellman Q，也不是模型预测 rollout。", font=font(15), fill=COLORS["muted"])
    image.save(ROOT / "assets/self_evolving_pipeline.png")


if __name__ == "__main__":
    render_method()
    render_evolution()
    print("rendered Chinese pipeline images")
