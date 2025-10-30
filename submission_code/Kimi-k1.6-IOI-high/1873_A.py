t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
    else:
        found = False
        # Check all possible single swaps
        for i in range(3):
            for j in range(i+1, 3):
                # Create a list to swap characters
                lst = list(s)
                lst[i], lst[j] = lst[j], lst[i]
                if ''.join(lst) == 'abc':
                    found = True
                    break
            if found:
                break
        print("YES" if found else "NO")