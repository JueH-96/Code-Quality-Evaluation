def solve():
    n = int(input())
    h = list(map(int, input().split()))

    max_count = 1
    for i in range(n):
        for j in range(i + 1, n):
            if h[i] == h[j]:
                count = 2
                diff = j - i
                for k in range(j + diff, n, diff):
                    if h[k] == h[i]:
                        count += 1
                    else:
                        break
                max_count = max(max_count, count)

    print(max_count)

solve()