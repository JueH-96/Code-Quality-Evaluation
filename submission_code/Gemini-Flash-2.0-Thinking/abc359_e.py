def solve():
    n = int(input())
    h = list(map(int, input().split()))

    results = []
    a = [0] * (n + 1)
    operations = 0

    first_positive = [float('inf')] * n

    while True:
        operations += 1
        a[0] += 1

        changed = False
        for i in range(1, n + 1):
            if a[i-1] > a[i] and a[i-1] > h[i-1]:  # Note: H is 0-indexed here
                a[i-1] -= 1
                a[i] += 1
                changed = True

        for i in range(n):
            if a[i+1] > 0 and first_positive[i] == float('inf'):
                first_positive[i] = operations

        if all(fp != float('inf') for fp in first_positive):
            break

        # Optional: Add a safeguard against infinite loops if needed, though unlikely given the problem constraints.
        if operations > 2 * 10**7:  # A large enough number based on constraints
            break

    print(*first_positive)

if __name__ == "__main__":
    solve()