def solve():
    n, s = map(int, input().split())
    a = list(map(int, input().split()))

    for length in range(1, 3 * n + 1):
        for start in range(3 * n - length + 1):
            current_sum = 0
            for i in range(length):
                current_sum += a[(start + i) % n]
            if current_sum == s:
                print("Yes")
                return

    print("No")

solve()