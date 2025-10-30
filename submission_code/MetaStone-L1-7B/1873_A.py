t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
        continue
    possible = False
    # Swap 0 and 1
    temp = list(s)
    temp[0], temp[1] = temp[1], temp[0]
    if ''.join(temp) == 'abc':
        possible = True
    # Swap 0 and 2
    temp = list(s)
    temp[0], temp[2] = temp[2], temp[0]
    if ''.join(temp) == 'abc':
        possible = True
    # Swap 1 and 2
    temp = list(s)
    temp[1], temp[2] = temp[2], temp[1]
    if ''.join(temp) == 'abc':
        possible = True
    print("YES" if possible else "NO")