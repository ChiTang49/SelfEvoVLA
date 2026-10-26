import csv
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NAV = [
    ("index.html", "概览"),
    ("method.html", "方法"),
    ("results.html", "实验结果"),
    ("demo.html", "视频演示"),
    ("data.html", "实验数据"),
]


def inline(text):
    parts = re.split(r"(`[^`]*`)", text)
    rendered = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            rendered.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        value = html.escape(part)
        value = re.sub(r"!\[([^]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', value)
        value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', value)
        value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="data.html#datasets">\1</a>', value)
        value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
        rendered.append(value)
    return "".join(rendered)


def table_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_to_html(text):
    lines = text.splitlines()
    output = []
    index = 0
    skip_level = None
    while index < len(lines):
        line = lines[index]
        if line.startswith("DO_NOT_PUBLISH"):
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            level = len(heading.group(1))
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if heading.group(2).startswith("DO_NOT_PUBLISH"):
                skip_level = level
                index += 1
                continue
            if skip_level is not None:
                index += 1
                continue
            output.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            index += 1
            continue
        if skip_level is not None:
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        if line.strip().startswith("```"):
            language = line.strip()[3:]
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            output.append(f'<pre><code class="language-{html.escape(language)}">{html.escape(chr(10).join(code))}</code></pre>')
            continue
        if line.strip().startswith("|") and index + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+", lines[index + 1]):
            headers = table_row(line)
            index += 2
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(table_row(lines[index]))
                index += 1
            output.append('<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{inline(cell)}</th>" for cell in headers) + "</tr></thead><tbody>")
            for row in rows:
                output.append("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>")
            output.append("</tbody></table></div>")
            continue
        if line.strip().startswith(">"):
            quote = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].strip())
                index += 1
            output.append(f"<blockquote>{inline(' '.join(quote))}</blockquote>")
            continue
        item = re.match(r"^\s*([-*+]|\d+\.)\s+(.+)", line)
        if item:
            ordered = bool(re.match(r"^\s*\d+\.", line))
            tag = "ol" if ordered else "ul"
            entries = []
            while index < len(lines):
                item = re.match(r"^\s*([-*+]|\d+\.)\s+(.+)", lines[index])
                if not item or bool(re.match(r"^\s*\d+\.", lines[index])) != ordered:
                    break
                entries.append(f"<li>{inline(item.group(2))}</li>")
                index += 1
            output.append(f"<{tag}>" + "".join(entries) + f"</{tag}>")
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip():
            next_line = lines[index]
            if re.match(r"^(#{1,6})\s+", next_line) or next_line.strip().startswith(("```", ">", "|")) or re.match(r"^\s*([-*+]|\d+\.)\s+", next_line):
                break
            paragraph.append(next_line.strip())
            index += 1
        output.append(f"<p>{inline(' '.join(paragraph))}</p>")
    return "\n".join(output)


def page(filename, active, title, lead, content):
    navigation = "".join(f'<a href="{href}" class="{"active" if href == active else ""}">{label}</a>' for href, label in NAV)
    document = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="{html.escape(lead)}">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="assets/site.css">
</head>
<body>
<header class="topbar"><nav class="nav" aria-label="页面导航"><a class="brand" href="index.html">SelfEvoVLA</a>{navigation}</nav></header>
<div class="hero"><div class="wrap"><div class="kicker">SelfEvoVLA · 开源研究项目</div><h1>{html.escape(title)}</h1><p class="lead">{html.escape(lead)}</p></div></div>
<main class="wrap">{content}</main>
<footer class="footer"><div class="wrap">基于 RIPT-VLA、QueST 与 LIBERO。实验过程、正负结果和适用边界均在页面中呈现。</div></footer>
</body>
</html>
'''
    (ROOT / filename).write_text(document, encoding="utf-8", newline="\n")


def video(source, poster, title, description):
    return f'''<div class="video-shell"><h3>{html.escape(title)}</h3><video controls preload="metadata" poster="{poster}"><source src="{source}" type="video/mp4">当前浏览器无法播放此视频。</video><p>{html.escape(description)}</p></div>'''


def csv_table(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = ['<div class="table-wrap table-scroll"><table><thead><tr>']
    output.extend(f"<th>{label}</th>" for _, label in columns)
    output.append("</tr></thead><tbody>")
    for row in rows:
        output.append("<tr>")
        for key, _ in columns:
            value = row[key]
            if key == "success":
                success = value.lower() == "true"
                value = "成功" if success else "失败"
                value = f'<span class="{"good" if success else "bad"}">{value}</span>'
            elif key == "method":
                value = {"sft": "SFT", "rloo": "RLOO", "e3": "有界 RIPT"}.get(value, value)
            elif value == "":
                value = "—"
            else:
                value = html.escape(value)
            output.append(f"<td>{value}</td>")
        output.append("</tr>")
    output.append("</tbody></table></div>")
    return "".join(output), len(rows)


TASK_LABELS = {"Main": "主任务", "Task 5": "任务 5", "Task 6": "任务 6"}


def result_table(data, key):
    labels = {"SFT": "SFT", "RIPT": "有界 RIPT", "SFT+Plan": "SFT＋规划", "RIPT+Plan": "RIPT＋规划"}
    rows = ['<div class="table-wrap"><table><thead><tr><th>任务</th>' + "".join(f"<th>{labels[method]}</th>" for method in labels) + "</tr></thead><tbody>"]
    for task, payload in data["tasks"].items():
        rows.append(f"<tr><td>{TASK_LABELS.get(task, task)}</td>")
        for method in labels:
            value = payload["methods"][method][key]
            rows.append(f'<td>{value["successes"]}/{value["episodes"]}（{value["success_rate"]:.1%}）</td>')
        rows.append("</tr>")
    rows.append('<tr class="highlight"><td>宏平均</td>')
    for method in labels:
        value = data["macro_average"][method][f"{key}_success_rate"]
        rows.append(f"<td><strong>{value:.1%}</strong></td>")
    rows.append("</tr></tbody></table></div>")
    return "".join(rows)


def value_table(data):
    output = ['<div class="table-wrap"><table><thead><tr><th>任务</th><th>AUROC</th><th>AUPRC</th><th>Brier</th></tr></thead><tbody>']
    for task, payload in data["tasks"].items():
        metrics = payload["value_metrics"]["test"]
        output.append(f'<tr><td>{TASK_LABELS.get(task, task)}</td><td>{metrics["auroc"]:.4f}</td><td>{metrics["auprc"]:.4f}</td><td>{metrics["brier"]:.4f}</td></tr>')
    output.append("</tbody></table></div>")
    return "".join(output)


def build():
    multitask = json.loads((ROOT / "evidence/multitask_summary.json").read_text(encoding="utf-8"))
    feedback = json.loads((ROOT / "evidence/feedback_summary.json").read_text(encoding="utf-8"))

    home = '''
<section><div class="section-head"><div class="kicker">01 · 项目要解决的问题</div><h2>让 VLA 不只模仿示范，还能从交互结果中改进</h2><p>行为克隆只能从已有示范学习。这个项目关注三个更具体的问题：策略能否利用成功与失败轨迹继续训练，动态采样能否避免无上限重试，以及结果价值模型能否帮助推理时的动作选择。</p></div>
<div class="grid three">
  <div class="card"><h3>交互后训练</h3><p>从 SFT 策略出发，在 LIBERO 中采样多条轨迹，用任务奖励计算组内优势并执行 RLOO/PPO 更新。</p></div>
  <div class="card"><h3>可控采样</h3><p>针对全成功或全失败轨迹组造成的零优势与重复采样，增加最大重试、预算耗尽保护和诊断日志。</p></div>
  <div class="card"><h3>自进化闭环</h3><p>进一步学习任务专属结果价值模型，对候选动作块排序，并把仿真验证成功的经验用于一轮回流。</p></div>
</div></section>
<section><div class="section-head"><div class="kicker">02 · 技术路线</div><h2>训练时后训练、推理时候选规划、验证经验回流</h2><p>QueST 主干在价值学习阶段保持冻结。候选动作在真实观测上生成，未选候选不进入仿真器；回流只使用仿真确认成功的轨迹，并限制为一轮。</p></div>
<figure class="figure"><img src="assets/self_evolving_pipeline.png" alt="SFT、RIPT、交互数据、结果价值、候选规划、执行验证与回流组成的闭环"></figure>
<div class="button-row"><a class="button" href="method.html">查看方法与技术细节</a></div></section>
<section><div class="section-head"><div class="kicker">03 · 核心实验结果</div><h2>真实训练得到的结果，也保留没有奏效的模块</h2><p>三个 LIBERO-Long 任务均完成真实 RIPT 训练、50 状态独立评估、任务专属价值模型和规划对照。</p></div>
<div class="grid four">
  <div class="card emphasis"><div class="label">RIPT 三任务宏平均</div><div class="metric">74.7<small>%</small></div><p>SFT 为 61.3%，提高 13.3 个百分点。</p></div>
  <div class="card emphasis"><div class="label">Task 6</div><div class="metric">88<small>%</small></div><p>SFT 为 56%，单训练 seed 提高 32 个百分点。</p></div>
  <div class="card"><div class="label">回流＋规划</div><div class="metric">72<small>%</small></div><p>两个评估 seed 均值，原规划为 66%。</p></div>
  <div class="card"><div class="label">价值模型</div><div class="metric">0.952<small>AUROC</small></div><p>三个任务留出 AUROC 为 0.789–0.952。</p></div>
</div>
<div class="note"><strong>需要同时看到的负结果：</strong>RIPT＋Planning 的三任务宏平均为 71.3%，低于 RIPT；三个任务的价值留出状态上，规划也均未改善。回流策略单独使用的两轮均值为 63%，低于原 E3 的 67%。</div>
<div class="button-row"><a class="button" href="results.html">查看完整结果矩阵</a><a class="button secondary" href="demo.html">观看真实行为对比</a></div></section>
<section><div class="section-head"><div class="kicker">04 · 可复现性</div><h2>不仅呈现成功率，也保留完整的实验数据链</h2></div>
<div class="grid three">
  <div class="card"><h3>真实训练与重载</h3><p>三个任务均接受 120 组、1,920 条有效训练轨迹；最终 checkpoint 均实际加载运行，而非只评估官方权重。</p></div>
  <div class="card"><h3>受控比较</h3><p>固定任务、初始状态与评估协议，分别给出全量 50 状态和最后 10 个价值留出状态，避免混用不同口径。</p></div>
  <div class="card"><h3>失败也可追溯</h3><p>保留共同失败和训练后退化；CSV、视频、图表与研究笔记均在页面中直接展示。</p></div>
</div></section>
<section><div class="section-head"><div class="kicker">05 · 页面导览</div><h2>从方法到实验数据</h2></div>
<div class="toc">
  <a href="method.html"><strong>方法</strong><span>先了解 RIPT、价值规划与回流如何连接</span></a>
  <a href="results.html"><strong>实验结果</strong><span>再看三任务、留出状态和两轮评估数字</span></a>
  <a href="demo.html"><strong>视频演示</strong><span>直观比较同一初始状态下的行为变化</span></a>
  <a href="data.html"><strong>实验数据</strong><span>最后检查 CSV、失败案例和研究笔记</span></a>
</div>
<div class="note"><strong>结论边界：</strong>每个任务使用一个策略训练 seed，主任务回流另有两个评估 seed。结果不代表跨训练 seed 稳定性、完整论文复现、SOTA 或真机效果。结果价值模型是学习到的 Q 代理，不是精确 Bellman Q 函数。</div></section>
'''
    page("index.html", "index.html", "自进化 VLA：从交互后训练到结果价值规划", "在单张 32GB GPU 上，完成 LIBERO 在线强化后训练、结果价值引导与验证经验回流，并用三任务实验检验效果和边界。", home)

    method = '''
<section><div class="section-head"><div class="kicker">01 · 官方基础</div><h2>RIPT：用组内相对表现更新 VLA 策略</h2><p>同一初始状态采样 K 条轨迹，使用留一法估计优势，不需要额外价值网络。全成功或全失败组的相对优势为零，因此需要在采样阶段识别并处理。</p></div>
<figure class="figure"><img src="assets/method_pipeline.png" alt="SFT、并行 rollout、组内优势、有界采样、PPO 与评估流程"></figure></section>
<section><div class="section-head"><div class="kicker">02 · 本项目扩展</div><h2>从后训练到自进化闭环</h2><p>本项目在官方 RIPT-VLA 基础上增加单卡实验配置、有界动态采样、结果价值规划、经验回流和分层评估证据。</p></div>
<figure class="figure"><img src="assets/self_evolving_pipeline.png" alt="RIPT、价值学习、候选规划与验证经验回流"></figure>
<div class="grid two">
  <div class="card"><h3>有界动态采样</h3><p>设置最大采样次数，记录接受组、全成功/全失败丢弃组和终止原因；预算耗尽时不把无效样本用于 PPO。</p></div>
  <div class="card"><h3>结果价值评估器</h3><p>联合上下文均值、当前观测和有序动作 token 表示，预测候选动作块的终端成功价值。QueST 主干冻结，只训练轻量 MLP。</p></div>
  <div class="card"><h3>候选动作规划</h3><p>复用官方随机采样路径，在同一真实观测上生成 K 个不同动作块，按结果价值评分选择执行，再进行下一次重规划。</p></div>
  <div class="card"><h3>验证经验回流</h3><p>只用仿真确认成功的规划轨迹微调策略先验；失败轨迹用于价值学习或失败分析，不进行无差别自我蒸馏。</p></div>
</div></section>
<section><div class="section-head"><div class="kicker">03 · 设计取舍</div><h2>为什么不直接复用“模型分数高”的结论</h2></div>
<div class="grid three">
  <div class="card"><h3>状态隔离</h3><p>价值数据按初始状态 0–29 训练、30–39 验证、40–49 留出。规划结果同时给出全量与留出状态，防止把见过的状态当作泛化。</p></div>
  <div class="card"><h3>效果与成本</h3><p>同时记录成功率、轨迹长度、采样丢弃、rollout 时间和候选推理耗时。并发条件下的耗时不包装成严格加速比。</p></div>
  <div class="card"><h3>结果价值并非精确 Q</h3><p>模型由终端 0/1 结果训练，没有显式 Bellman 备份，也不对未选候选执行模型预测 rollout，因此称为结果价值或 Q 代理。</p></div>
</div></section>
<section><div class="section-head"><div class="kicker">04 · 工程实现</div><h2>单卡运行与可复现实现</h2></div>
<div class="grid two">
  <div class="card"><h3>单卡与并行 rollout</h3><p>使用 32GB GPU 完成训练。8 并行环境首轮采样较 4 并行减少 26.62%，但该数字只针对首轮，不代表总训练加速。</p></div>
  <div class="card"><h3>断点与配置验收</h3><p>实现初始状态偏移续采、参数覆盖 dry-run 和 checkpoint 恢复；失败实验不覆盖原目录，也不将中间权重当作最终结果。</p></div>
  <div class="card"><h3>完整记录</h3><p>保存配置、日志、模型哈希、逐 episode 结果和视频。CSV、实验文档和图表可直接在本网站检查。</p></div>
  <div class="card"><h3>适用边界</h3><p>实验限于 LIBERO-Long、QueST 和单策略训练 seed，不扩展到 RoboTwin、真机或 7B OpenVLA-OFT。</p></div>
</div></section>
'''
    page("method.html", "method.html", "方法：强化后训练与结果价值引导", "先解释为什么这样设计，再判断模块是否有效。官方 RIPT 基础与本项目扩展在页面中明确区分。", method)

    overall = result_table(multitask, "overall")
    heldout = result_table(multitask, "heldout")
    values = value_table(multitask)
    feedback_rows = []
    feedback_names = {
        "E3, K=1": "原 E3，K=1",
        "Feedback, K=1": "回流，K=1",
        "E3 + Planning, K=4": "原 E3＋规划，K=4",
        "Feedback + Planning, K=4": "回流＋规划，K=4",
    }
    for name, payload in feedback["methods"].items():
        feedback_rows.append(f"<tr><td>{feedback_names.get(name, name)}</td><td>{payload['seeds'][0]['overall']['success_rate']:.0%}</td><td>{payload['seeds'][1]['overall']['success_rate']:.0%}</td><td><strong>{payload['mean_overall_success_rate']:.1%}</strong></td><td>{payload['mean_heldout_success_rate']:.1%}</td></tr>")
    results = f'''
<section><div class="section-head"><div class="kicker">01 · 三任务主结果</div><h2>RIPT 宏平均从 61.3% 提升至 74.7%</h2><p>每个任务使用一个策略训练 seed 和 50 个固定初始状态。主任务采用本轮 K=1 受控口径，与早期单任务历史评估分开。</p></div>{overall}
<figure class="figure"><img src="assets/multitask_results.png" alt="三任务全量、留出状态和价值指标图"></figure></section>
<section><div class="section-head"><div class="kicker">02 · 留出状态</div><h2>规划的收益没有迁移到价值留出状态</h2><p>最后 10 个初始状态不参与价值模型训练或回流。三个任务的 RIPT＋Planning 均低于对应 RIPT，不能选择性使用 Task 5 的全量 78% 作为普遍结论。</p></div>{heldout}</section>
<section><div class="section-head"><div class="kicker">03 · 一轮经验回流</div><h2>组合方法改善，回流单独使用并未改善</h2><p>回流使用 21 条仿真确认成功的规划轨迹、843 个动作块，微调 2 个 epoch。下表同时给出两个评估 seed 和留出均值。</p></div>
<div class="table-wrap"><table><thead><tr><th>模型与方式</th><th>seed 0</th><th>seed 1</th><th>全量均值</th><th>留出均值</th></tr></thead><tbody>{''.join(feedback_rows)}</tbody></table></div>
<figure class="figure"><img src="assets/feedback_results.png" alt="原 E3 与回流模型在两个评估 seed 下的结果"></figure></section>
<section><div class="section-head"><div class="kicker">04 · 价值模型与规划消融</div><h2>排序能力存在，但不等于在线规划收益</h2><p>三个任务专属价值模型均有可测量的留出排序能力；K=1/4/8 对照用于分析候选数量和实际执行收益。</p></div>
<div class="grid two"><div>{values}</div><div><figure class="figure"><img src="assets/planning_results.png" alt="K=1、4、8 全量、留出结果和推理耗时"></figure></div></div>
<figure class="figure"><img src="assets/candidate_scores.png" alt="前 24 次重规划的候选评分与选择"></figure></section>
<section><div class="section-head"><div class="kicker">05 · 训练过程</div><h2>训练曲线与采样量同时公开</h2><p>训练期 rollout 成功率只用于观察过程，不替代最终独立评估。蓝色为接受轨迹，橙色为动态采样丢弃轨迹。</p></div>
<div class="grid three">
  <figure class="figure"><img src="evidence/E3-main-bript-seed00/training_curve.png" alt="主任务 RIPT 训练曲线"><figcaption>主任务：15 轮 RIPT</figcaption></figure>
  <figure class="figure"><img src="evidence/E11-task5-bript-resume4-v2/training_curve.png" alt="Task 5 恢复段训练曲线"><figcaption>Task 5：原 5 轮＋恢复 10 轮中的恢复段</figcaption></figure>
  <figure class="figure"><img src="evidence/E11-task6-bript-final-seed00/training_curve.png" alt="Task 6 训练曲线"><figcaption>Task 6：15 轮 RIPT</figcaption></figure>
</div></section>
<section><div class="section-head"><div class="kicker">06 · 如何理解这些数字</div><h2>结果可以支持什么，不能支持什么</h2></div>
<div class="grid three">
  <div class="card"><h3>可以支持</h3><p>单卡完成真实 RIPT；两个扩展任务明显提升；价值模型具有留出排序能力；回流与规划组合在两个评估 seed 上呈正向。</p></div>
  <div class="card"><h3>需要保留</h3><p>规划在三任务留出状态均未改善；回流单独使用退化；评估存在波动；Task 5 为两阶段恢复训练。</p></div>
  <div class="card"><h3>不能外推</h3><p>不能宣称跨训练 seed 稳定提升、完整 Q-Planning 复现、SOTA、RoboTwin 或真机效果。</p></div>
</div></section>
'''
    page("results.html", "results.html", "实验结果：有效提升与无效模块都如实呈现", "三任务结果、留出状态、两轮回流、K 消融和训练曲线集中展示，便于快速核对数字与结论。", results)

    paired, paired_count = csv_table(ROOT / "evidence/paired_episode_results.csv", [
        ("method", "方法"), ("init_state_index", "初始状态"), ("success", "结果"),
        ("episode_steps", "环境步数"), ("failure_type", "失败类型"), ("observed_evidence", "观察依据"),
    ])
    evolution, evolution_count = csv_table(ROOT / "assets/evolution_cases.csv", [
        ("experiment_id", "实验编号"), ("init_state_index", "初始状态"), ("success", "结果"), ("episode_steps", "环境步数"),
    ])
    demo = f'''
<section><div class="section-head"><div class="kicker">01 · 三方法同步对比</div><h2>相同初始状态，不只保留成功案例</h2><p>SFT、RLOO 和有界 RIPT 同步播放。75 秒内容包含共同失败、训练后改善和训练后退化；视频只用于定性分析，不替代正式 50 状态评估。</p></div>
{video("assets/ript_comparison_demo.mp4", "assets/demo_cover.png", "SFT / RLOO / 有界 RIPT", "初始状态 2：共同失败；初始状态 4：两种自训方法改善；初始状态 0：SFT 成功但自训方法失败。")}</section>
<section><div class="section-head"><div class="kicker">02 · 训练与自进化案例</div><h2>把 checkpoint 和行为变化对应起来</h2><p>每个案例均标注实际初始状态和环境步数。随机策略在其他 rollout 中可能表现不同，因此不将精选视频当作稳定性证明。</p></div>
{video("assets/evolution_checkpoint.mp4", "assets/evolution_checkpoint_cover.png", "训练迭代：第 5 轮失败 → 第 15 轮成功", "主任务、初始状态 47：RIPT 第 5 轮在 600 步失败，第 15 轮在 280 步成功。")}
{video("assets/evolution_feedback.mp4", "assets/evolution_feedback_cover.png", "回流＋规划：原 E3 失败 → 组合方法成功", "主任务、初始状态 7：原 E3 在 600 步失败，回流＋K=4 规划在 359 步成功。")}
{video("assets/evolution_new_tasks.mp4", "assets/evolution_new_tasks_cover.png", "跨任务：SFT 失败 → RIPT 成功", "Task 5 初始状态 16、Task 6 初始状态 2：SFT 均在 600 步失败，RIPT 分别在 257 和 243 步成功。")}</section>
<section><div class="section-head"><div class="kicker">03 · 失败案例分析</div><h2>失败不是统一的一类问题</h2><p>标签来自单视角视频的人工观察，置信度为中，不冒充仿真器自动真值。没有足够证据时不强行归类。</p></div>
<div class="table-wrap"><table><thead><tr><th>方法</th><th>未抓取</th><th>放置失败</th><th>中途停滞</th><th>合计失败</th></tr></thead><tbody>
<tr><td>SFT</td><td>0</td><td>1</td><td>1</td><td>2</td></tr>
<tr><td>RLOO</td><td>1</td><td>2</td><td>0</td><td>3</td></tr>
<tr><td>有界 RIPT</td><td>0</td><td>2</td><td>2</td><td>4</td></tr>
</tbody></table></div>
<div class="grid three">
  <div class="card"><h3>共同失败：状态 2</h3><p>三种方法都达到 600 步上限，提示存在共同困难状态，不能只看平均成功率。</p></div>
  <div class="card"><h3>训练后改善：状态 4</h3><p>SFT 失败，RLOO 和有界 RIPT 成功，说明策略更新可改变具体行为。</p></div>
  <div class="card"><h3>训练后退化：状态 0 和 3</h3><p>SFT 成功而至少一种自训方法失败，这些案例同样保留在页面中。</p></div>
</div></section>
<section><div class="section-head"><div class="kicker">04 · 案例记录</div><h2>{evolution_count} 条生成案例完整呈现</h2><p>除页面精选视频外，同时保留未改善和退化记录。完整逐 episode 数据还可在实验数据页查看。</p></div>{evolution}</section>
'''
    page("demo.html", "demo.html", "视频演示：同一初始状态下的真实行为变化", "用大尺寸视频比较 SFT、RLOO、RIPT、规划与回流，同时展示改善、共同失败和退化案例。", demo)

    evidence = f'''
<section><div class="section-head"><div class="kicker">01 · 逐 episode 数据</div><h2>{paired_count} 条配对轨迹直接呈现在页面中</h2><p>三种方法各 8 条，包含成功、失败、环境步数、失败类型和观察依据。这里只用于定性分析，不用小样本重新排名。</p></div>{paired}</section>
<section id="datasets"><div class="section-head"><div class="kicker">02 · 数据组成</div><h2>训练、验证与留出数据各自承担什么作用</h2><p>数据只用于解释实验设置和结果，不包含原始执行日志或服务器环境信息。</p></div>
<div class="grid three">
  <div class="card"><h3>价值学习数据</h3><p>主任务使用 SFT 与 E3 各 50 条轨迹，共 7,350 个动作块；状态 0–29 训练、30–39 验证、40–49 留出。</p></div>
  <div class="card"><h3>回流数据</h3><p>一轮回流使用 21 条仿真确认成功的规划轨迹、843 个动作块，仅来自训练状态范围。</p></div>
  <div class="card"><h3>结果摘要</h3><p>三任务和回流结果由结构化摘要生成，保留全量与留出数字，不用精选视频替代正式评估。</p></div>
</div></section>
<section><div class="section-head"><div class="kicker">03 · 可复现性检查</div><h2>每个核心数字都能回到实验数据</h2></div>
<div class="grid three">
  <div class="card"><h3>模型与配置</h3><p>最终 RIPT、回流和价值模型 checkpoint 均保留，关键运行包含配置覆盖项、训练 seed 和 checkpoint 哈希。</p></div>
  <div class="card"><h3>数据划分</h3><p>价值训练、验证和留出状态分离；回流仅使用训练状态内的成功轨迹，不将留出状态用于微调。</p></div>
  <div class="card"><h3>科学结论边界</h3><p>保留规划退化和回流单独使用退化，明确这些方法在哪些条件下没有奏效。</p></div>
</div></section>
'''
    page("data.html", "data.html", "实验数据：轨迹、失败案例与可复现性", "CSV 与数据说明直接渲染在页面中，便于逐项核对方法、结果、失败类型和实验边界。", evidence)
    print("built index.html, method.html, results.html, demo.html, data.html")


if __name__ == "__main__":
    build()
