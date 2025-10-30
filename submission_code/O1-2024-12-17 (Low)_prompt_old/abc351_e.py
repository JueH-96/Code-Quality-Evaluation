def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = input_data[1::2]
    Y = input_data[2::2]

    # We'll separate the points into two sets according to (x+y) mod 2.
    # The distance between two points with different (x+y) mod 2 is 0 (unreachable).
    # If (x+y) mod 2 is the same, we map (x,y) to (u,v) = ((x+y)//2, (x-y)//2)
    # and the distance in the original problem is the L1 distance between these (u,v) pairs.

    S0u, S0v = [], []  # for points where (x+y) mod 2 == 0
    S1u, S1v = [], []  # for points where (x+y) mod 2 == 1

    idx = 0
    for _ in range(N):
        x = int(X[idx])
        y = int(Y[idx])
        idx += 1
        if ((x + y) & 1) == 0:
            # (x+y) even
            S0u.append((x + y) // 2)
            S0v.append((x - y) // 2)
        else:
            # (x+y) odd
            S1u.append((x + y) // 2)
            S1v.append((x - y) // 2)

    # Function to compute sum of pairwise absolute differences in a sorted array
    def pairwise_abs_diff_sum(arr):
        arr.sort()
        prefix_sum = 0
        total = 0
        for i, val in enumerate(arr):
            total += val * i - prefix_sum
            prefix_sum += val
        return total

    ans = 0
    # sum of pairwise L1 distances = sum of pairwise |u_i - u_j| + sum of pairwise |v_i - v_j|
    if len(S0u) > 1:
        ans += pairwise_abs_diff_sum(S0u) + pairwise_abs_diff_sum(S0v)
    if len(S1u) > 1:
        ans += pairwise_abs_diff_sum(S1u) + pairwise_abs_diff_sum(S1v)

    print(ans)

# Let's call solve() to execute
# (The testing environment will call solve() itself, but we'll call it once here
#  so that the code is self-contained if run directly.)
def main():
    solve()

if __name__ == "__main__":
    main()