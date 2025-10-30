t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
        continue
    possible = False
    s_list = list(s)
    swaps = [(0, 1), (0, 2), (1, 2)]
    for i, j in swaps:
        temp = s_list.copy()
        temp[i], temp[j] = temp[j], temp[i]
        if ''.join(temp) == 'abc':
            possible = True
            break
    print("YES" if possible else "NO")