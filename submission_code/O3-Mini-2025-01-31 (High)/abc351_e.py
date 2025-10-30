def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # We separate the points into two groups based on the parity of (x+y).
    # A rabbit can only move diagonally and every jump preserves the parity of (x+y).
    # Therefore, if two points have different parity, the distance is defined to be 0.
    # For points in the same parity group, the minimal number of jumps needed from
    # (x1, y1) to (x2, y2) is max(|x2-x1|, |y2-y1|).
    # We can show that:
    #   max(|x2-x1|, |y2-y1|) = (| (x1+y1) - (x2+y2) | + | (x1-y1) - (x2-y2) |) // 2.
    #
    # Let u = x+y and v = x-y. Then for each same-parity group we want to compute:
    #   Sum_{i<j} max(|dx|,|dy|) = 1/2 * ( Sum_{i<j} |u_i-u_j| + Sum_{i<j} |v_i-v_j| ).
    #
    # The sums of absolute differences over pairs can be computed efficiently by sorting.
    
    group0_u = []  # for points where (x+y) is even
    group0_v = []
    group1_u = []  # for points where (x+y) is odd
    group1_v = []
    
    for _ in range(n):
        x = int(next(it))
        y = int(next(it))
        if ((x + y) & 1) == 0:
            group0_u.append(x + y)
            group0_v.append(x - y)
        else:
            group1_u.append(x + y)
            group1_v.append(x - y)
    
    # Helper function to compute sum_{i<j}|a[j]-a[i]| given an array a.
    # When the array is sorted, this sum equals:
    #   sum_{j=0}^{m-1} a[j]*(2*j - m + 1)
    def pair_diff_sum(arr):
        arr.sort()
        m = len(arr)
        total = 0
        for j, val in enumerate(arr):
            total += val * (2 * j - m + 1)
        return total

    total_sum = 0
    
    # Process group0 (even parity group)
    if len(group0_u) > 1:
        sum_u_diff = pair_diff_sum(group0_u)
        sum_v_diff = pair_diff_sum(group0_v)
        # Each pair’s contribution is (|u_i-u_j| + |v_i-v_j|) // 2.
        total_sum += (sum_u_diff + sum_v_diff) // 2
    
    # Process group1 (odd parity group)
    if len(group1_u) > 1:
        sum_u_diff = pair_diff_sum(group1_u)
        sum_v_diff = pair_diff_sum(group1_v)
        total_sum += (sum_u_diff + sum_v_diff) // 2
    
    sys.stdout.write(str(total_sum))

if __name__ == '__main__':
    main()