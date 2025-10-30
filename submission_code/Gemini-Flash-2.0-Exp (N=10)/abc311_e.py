def solve():
    h, w, n = map(int, input().split())
    holes = set()
    for _ in range(n):
        a, b = map(int, input().split())
        holes.add((a, b))

    count = 0
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            for size in range(1, min(h - i + 2, w - j + 2)):
                is_holeless = True
                for row_offset in range(size):
                    for col_offset in range(size):
                        if (i + row_offset, j + col_offset) in holes:
                            is_holeless = False
                            break
                    if not is_holeless:
                        break
                if is_holeless:
                    count += 1
    print(count)

solve()