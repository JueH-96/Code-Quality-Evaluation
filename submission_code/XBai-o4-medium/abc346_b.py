W, B = map(int, input().split())
pattern = "wbwbwwbwbwbw"
total = W + B
found = False

if total == 0:
    print("No")
else:
    for start in range(12):
        w_count = 0
        for i in range(total):
            pos = start + i
            if pattern[pos % 12] == 'w':
                w_count += 1
        if w_count == W:
            found = True
            break
    print("Yes" if found else "No")