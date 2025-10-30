def solve():
    n = int(input())
    a = list(map(int, input().split()))

    b = []
    for i in range(n):
        weekly_sum = 0
        for j in range(7):
            weekly_sum += a[i * 7 + j]
        b.append(weekly_sum)

    print(*b)

if __name__ == "__main__":
    solve()