def solve():
    H, W, N = map(int, input().split())
    holes = []
    for _ in range(N):
        holes.append(tuple(map(int, input().split())))

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