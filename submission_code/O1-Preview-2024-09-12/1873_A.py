# YOUR CODE HERE
t = int(input())
for _ in range(t):
    s = input()
    if s == 'abc':
        print('YES')
    else:
        found = False
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            s_list = list(s)
            s_list[i], s_list[j] = s_list[j], s_list[i]
            s_swapped = ''.join(s_list)
            if s_swapped == 'abc':
                found = True
                break
        if found:
            print('YES')
        else:
            print('NO')