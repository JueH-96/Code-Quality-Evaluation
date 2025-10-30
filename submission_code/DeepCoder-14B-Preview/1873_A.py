t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
        continue
    possible = False
    # Check all possible swaps
    for i in range(3):
        for j in range(i+1, 3):
            lst = list(s)
            lst[i], lst[j] = lst[j], lst[i]
            if ''.join(lst) == 'abc':
                possible = True
                break
        if possible:
            break
    print("YES" if possible else "NO")