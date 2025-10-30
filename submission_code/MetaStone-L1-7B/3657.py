def check_horizontal(rectangles):
    end_ys = [r[1] for r in rectangles]
    start_ys = [r[0] for r in rectangles]
    max_end = max(end_ys)
    min_start = min(start_ys)
    if max_end >= min_start:
        for r in rectangles:
            if r[0] > max_end and r[1] < min_start:
                return True
        return False
    else:
        return True

def check_vertical(rectangles):
    end_xs = [r[2] for r in rectangles]
    start_xs = [r[0] for r in rectangles]
    max_end = max(end_xs)
    min_start = min(start_xs)
    if max_end >= min_start:
        for r in rectangles:
            if r[1] > max_end and r[0] < min_start:
                return True
        return False
    else:
        return True

def check_condition(n, rectangles):
    if check_horizontal(rectangles):
        return True
    if check_vertical(rectangles):
        return True
    return False

n = int(input())
rectangles = [tuple(map(int, input().split())) for _ in range(n)]
print(check_condition(n, rectangles))