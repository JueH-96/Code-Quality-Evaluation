def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, D, P = map(int, data[:3])
    F = list(map(int, data[3:]))

    # Sort fares in descending order
    F.sort(reverse=True)

    # Compute prefix sums: prefix_sum[i] = sum of F[0]..F[i-1]
    prefix_sum = [0] * (N+1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + F[i-1]

    # Try covering k days with passes (where k ranges from 0..N)
    # pass_count = ceil(k / D) => pass_count = (k + D - 1) // D
    # cost = pass_count * P + (prefix_sum[N] - prefix_sum[k])
    # We'll keep track of the minimum such cost.
    import math

    min_cost = float('inf')
    for k in range(N+1):
        pass_count = (k + D - 1) // D
        cost = pass_count * P + (prefix_sum[N] - prefix_sum[k])
        if cost < min_cost:
            min_cost = cost

    print(min_cost)

def main():
    solve()

if __name__ == "__main__":
    main()