def solve():
    n, k = map(int, input().split())
    s_list = list(input())
    operations_count = 0
    while True:
        last_black_index = -1
        for i in range(n):
            if s_list[i] == 'B':
                last_black_index = i
        if last_black_index == -1:
            break
        start_index = max(0, last_black_index - k + 1)
        end_index = last_black_index
        for i in range(start_index, end_index + 1):
            s_list[i] = 'W'
        operations_count += 1
    print(operations_count)

t = int(input())
for _ in range(t):
    solve()