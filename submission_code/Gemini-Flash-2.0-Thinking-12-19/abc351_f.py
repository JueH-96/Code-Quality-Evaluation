def solve():
    n = int(input())
    a = list(map(int, input().split()))
    total_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = a[j] - a[i]
            total_sum += max(diff, 0)
    print(total_sum)

if __name__ == '__main__':
    solve()