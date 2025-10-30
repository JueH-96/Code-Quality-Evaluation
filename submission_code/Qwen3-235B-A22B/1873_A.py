t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
        continue
    possible = False
    # Check all possible swaps
    swaps = [(0,1), (0,2), (1,2)]
    for i, j in swaps:
        lst = list(s)
        lst[i], lst[j] = lst[j], lst[i]
        new_s = ''.join(lst)
        if new_s == 'abc':
            possible = True
            break
    print("YES" if possible else "NO")