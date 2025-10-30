def solve():
    n, k = map(int, input().split())
    s_list = list(input())
    count = 0
    i = 0
    while i <= n - k:
        is_healthy = True
        for j in range(i, i + k):
            if s_list[j] == 'X':
                is_healthy = False
                break
        if is_healthy:
            count += 1
            for j in range(i, i + k):
                s_list[j] = 'X'
            i += k
        else:
            i += 1
    print(count)

if __name__ == "__main__":
    solve()