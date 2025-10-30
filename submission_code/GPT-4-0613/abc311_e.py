H, W, N = map(int, input().split())
holes = set(tuple(map(int, input().split())) for _ in range(N))

holeless_squares = 0
for i in range(1, H + 1):
    for j in range(1, W + 1):
        n = 1
        while i + n - 1 <= H and j + n - 1 <= W:
            if any((i + k, j + l) in holes for k in range(n) for l in range(n)):
                break
            holeless_squares += 1
            n += 1

print(holeless_squares)