from matplotlib.font_manager import fontManager
print([f.name for f in fontManager.ttflist if 'SimHei' in f.name])