t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
        continue
    chars = list(s)
    found = False
    # Check all possible single swaps
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        temp = chars.copy()
        temp[i], temp[j] = temp[j], temp[i]
        if ''.join(temp) == 'abc':
            found = True
            break
    print("YES" if found else "NO")