def solve():
    n, s = map(int, input().split())
    a = list(map(int, input().split()))

    extended_a = a * 2

    for i in range(2 * n):
        current_sum = 0
        for j in range(i, 2 * n):
            current_sum += extended_a[j]
            if current_sum == s:
                print("Yes")
                return

    print("No")

if __name__ == "__main__":
    solve()