def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    # We will separate points by parity of (x+y) % 2.
    # For each group, we transform each point (x,y) to (u,v) = (x+y, x-y).
    # It can be shown that the "rabbit‐jump" distance between two reachable points
    # is max(|dx|,|dy|) = (|du| + |dv|) // 2. Hence for each parity group,
    # sum(dist) = (sum_{pairs} |du| + sum_{pairs} |dv|) // 2.
    from collections import defaultdict
    groups = {0: [([], [])], 1: [([], [])]}  # we'll actually just use two lists

    u_lists = {0: [], 1: []}
    v_lists = {0: [], 1: []}

    for _ in range(N):
        x, y = map(int, input().split())
        p = (x + y) & 1
        u = x + y
        v = x - y
        u_lists[p].append(u)
        v_lists[p].append(v)

    ans = 0
    # helper to compute sum of abs differences in one list
    def sum_abs_diff(arr):
        # arr is list of ints, length m
        # sum_{i<j} |arr[i] - arr[j]| = 
        #   if sorted arr -> sum_{k=0..m-1} arr[k] * (2*k - m + 1)
        arr.sort()
        m = len(arr)
        total = 0
        # We can accumulate prefix sums or use the formula directly in one pass.
        # Let's do one pass:
        # total = sum_{k} arr[k] * (2*k - m + 1)
        # careful with large ints
        coeff_base = -(m - 1)
        # then at k we have arr[k] * (coeff_base + 2*k)
        for k, val in enumerate(arr):
            total += val * (coeff_base + 2 * k)
        return total

    for p in (0, 1):
        m = len(u_lists[p])
        if m <= 1:
            continue
        su = sum_abs_diff(u_lists[p])
        sv = sum_abs_diff(v_lists[p])
        # distance sum for this parity group:
        # (sum |du| + sum |dv|) // 2
        ans += (su + sv) // 2

    print(ans)

if __name__ == "__main__":
    main()