from itertools import product

def check(h, w, tiles):
    if h * w != sum(a * b for a, b in tiles):
        return False
    for layout in product((0, 1), repeat=len(tiles)):
        heights = [a if not l else b for (a, b), l in zip(tiles, layout)]
        widths = [b if not l else a for (a, b), l in zip(tiles, layout)]
        heights.sort(reverse=True)
        widths.sort(reverse=True)
        if len(heights) > h or len(widths) > w:
            continue
        if any((h - sum(heights[:i + 1])) * (w - sum(widths[:i + 1])) < 0 for i in range(len(tiles))):
            continue
        return True
    return False

n, h, w = map(int, input().split())
tiles = [tuple(map(int, input().split())) for _ in range(n)]
print('Yes' if any(check(h, w, tiles) for _ in range(2 ** n)) else 'No')