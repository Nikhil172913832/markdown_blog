import markdown
from pathlib import Path

posts_dir = Path("posts")
build_dir = Path("build")

build_dir.mkdir(exist_ok=True)
(build_dir / ".nojekyll").touch()

for md_file in posts_dir.glob("*.md"):
    html = markdown.markdown(md_file.read_text(encoding="utf-8"))
    
    # Wrap with basic HTML boilerplate
    html_page = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>{md_file.stem}</title></head>
<body>{html}</body>
</html>"""
    
    output_path = build_dir / f"{md_file.stem}.html"
    output_path.write_text(html_page, encoding="utf-8")

index_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>My Blog</title></head>
<body>
<h1>My Blog Posts</h1>
<ul>
{''.join(links)}
</ul>
</body>
</html>"""

(build_dir / "index.html").write_text(index_html, encoding="utf-8")