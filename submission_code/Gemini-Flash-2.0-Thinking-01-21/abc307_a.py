def solve():
    n = int(input())
    a = list(map(int, input().split()))

    b = []
    for i in range(n):
        start_index = i * 7
        end_index = (i + 1) * 7
        week_steps = sum(a[start_index:end_index])
        b.append(week_steps)

    print(*b)

if __name__ == "__main__":
    solve()