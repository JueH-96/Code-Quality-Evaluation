def solve():
    h, w, n = map(int, input().split())
    holes = set()
    for _ in range(n):
        r, c = map(int, input().split())
        holes.add((r, c))

    count = 0
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            max_n = 0
            for n_size in range(1, min(h - i + 2, w - j + 2)):
                is_holeless = True
                for row_offset in range(n_size):
                    for col_offset in range(n_size):
                        if (i + row_offset, j + col_offset) in holes:
                            is_holeless = False
                            break
                    if not is_holeless:
                        break
                if is_holeless:
                    max_n = n_size
                else:
                    break
            count += max_n
    print(count)

solve()