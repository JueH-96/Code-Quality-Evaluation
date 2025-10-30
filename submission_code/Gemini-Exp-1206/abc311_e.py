def solve():
    H, W, N = map(int, input().split())
    holes = set()
    for _ in range(N):
        a, b = map(int, input().split())
        holes.add((a, b))

    count = 0
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            for n in range(1, min(H - i + 2, W - j + 2)):
                is_holeless = True
                for k in range(n):
                    for l in range(n):
                        if (i + k, j + l) in holes:
                            is_holeless = False
                            break
                    if not is_holeless:
                        break
                if is_holeless:
                    count += 1

    print(count)

solve()