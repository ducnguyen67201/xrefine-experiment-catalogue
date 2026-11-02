"""Reflow the validated baseline workflow into a readable document figure.

This print derivative preserves both six-step chains and their insertion point.
The Archify workflow JSON remains the source-backed editable diagram.
"""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'docs/diagrams/baseline-vs-crossmatch.workflow.json').read_text())
assert len(data['nodes']) == 12 and len(data['edges']) == 10
for prefix in ('base', 'ours'):
    assert {(e['from'], e['to']) for e in data['edges'] if e['from'].startswith(prefix)} == {
        (prefix + str(i), prefix + str(i + 1)) for i in range(5)
    }

labels = [
    ('1. Same images and matches', '1. Same images and matches', 'Raw endpoints p0, p1'),
    ('2. Local patch features', '2. Same local patch features', '11x11 patches; CNN + within-match attention'),
    ('3. Frozen 3x3 score head', '3. ADD other-match context', None),
    ('4. Decode refined q0, q1', '4. Same coordinate decoder', 'Soft-argmax; retain all correspondence IDs'),
    ('5. Estimate camera pose', '5. Same pose pipeline', 'GC-RANSAC + pose recovery'),
    ('6. Measure accuracy / tails', '6. Same evaluation', 'Matched pilot AUC5 and severe errors'),
]

svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="730" viewBox="0 0 900 730" role="img" aria-labelledby="title desc">',
       '<title id="title">XRefine baseline versus our cross-match experiment</title>',
       '<desc id="desc">Two six-step chains keep the same input matches, local patch features, coordinate decoder, pose estimation and metrics. The experimental chain inserts other-match mean or learned attention and bounded channel modulation before the frozen score-map head. Independent and shuffled controls test the context association.</desc>',
       '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#64748b"/></marker></defs>',
       '<rect width="900" height="730" fill="white"/>',
       '<g font-family="Arial, sans-serif" fill="#17212f">',
       '<text x="450" y="30" text-anchor="middle" font-size="24" font-weight="700">Where we changed XRefine</text>',
       '<text x="220" y="63" text-anchor="middle" font-size="20" font-weight="700">BASELINE B</text>',
       '<text x="680" y="63" text-anchor="middle" font-size="20" font-weight="700">OUR CROSS-MATCH VARIANTS</text>']

for i, (base, ours, sub) in enumerate(labels):
    y = 82 + i * 84
    for x, label, changed in ((15, base, False), (475, ours, i == 2)):
        fill, stroke = ('#fff0d6', '#ba6a0a') if changed else ('#eef6fb', '#6687a0')
        svg.append(f'<rect x="{x}" y="{y}" width="410" height="66" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        svg.append(f'<text x="{x+16}" y="{y+25}" font-size="19" font-weight="700">{escape(label)}</text>')
        detail = sub
        if i == 2:
            detail = 'No module linking different matches' if not changed else 'Mean / attention -> channel scaling -> same head'
        svg.append(f'<text x="{x+16}" y="{y+49}" font-size="15">{escape(detail)}</text>')
        if i < 5:
            svg.append(f'<path d="M{x+205} {y+67}V{y+80}" stroke="#64748b" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')

svg += ['<rect x="15" y="600" width="870" height="111" rx="9" fill="#fafbfc" stroke="#cbd5e1"/>',
        '<text x="30" y="625" font-size="17" font-weight="700">What changed: context from eight other matches before the score head.</text>',
        '<text x="30" y="649" font-size="16">Independent control: own-match information only.</text>',
        '<text x="30" y="672" font-size="16">Shuffled control: real same-pair features; prescribed neighbor links broken.</text>',
        '<text x="30" y="695" font-size="16">All other stages shown are kept within each matched comparison.</text>',
        '</g></svg>']
target = ROOT / 'docs/diagrams/baseline-vs-crossmatch-print.svg'
target.write_text('\n'.join(svg) + '\n')
print(target)
