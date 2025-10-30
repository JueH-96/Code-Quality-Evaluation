def solve():
    n, k = map(int, input().split())
    s_str = input()
    s_list = list(s_str)
    operations = 0
    i = 0
    while i < n:
        if s_list[i] == 'B':
            operations += 1
            end_index = min(i + k - 1, n - 1)
            for j in range(i, end_index + 1):
                s_list[j] = 'W'
            i = end_index + 1
        else:
            i += 1
    print(operations)

t = int(input())
for _ in range(t):
    solve()