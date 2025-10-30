# Read the input
H, W, N = map(int, input().split())
holes = set()
for _ in range(N):
    a, b = map(int, input().split())
    holes.add((a, b))

# Count the number of holeless squares
count = 0
for n in range(1, min(H, W) + 1):
    for i in range(1, H - n + 2):
        for j in range(1, W - n + 2):
            is_holeless = True
            for k in range(i, i + n):
                for l in range(j, j + n):
                    if (k, l) in holes:
                        is_holeless = False
                        break
                if not is_holeless:
                    break
            if is_holeless:
                count += 1

print(count)