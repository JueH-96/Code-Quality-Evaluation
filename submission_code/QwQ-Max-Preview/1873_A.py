t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
        continue
    # Check all possible single swaps
    s_list = list(s)
    # Swap 0 and 1
    swapped = s_list.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    if ''.join(swapped) == 'abc':
        print("YES")
        continue
    # Swap 0 and 2
    swapped = s_list.copy()
    swapped[0], swapped[2] = swapped[2], swapped[0]
    if ''.join(swapped) == 'abc':
        print("YES")
        continue
    # Swap 1 and 2
    swapped = s_list.copy()
    swapped[1], swapped[2] = swapped[2], swapped[1]
    if ''.join(swapped) == 'abc':
        print("YES")
        continue
    # None worked
    print("NO")