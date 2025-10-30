def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    coords = [(int(input_data[2*i+1]), int(input_data[2*i+2])) for i in range(N)]
    
    # Separate points by parity of x+y
    even_points = []
    odd_points = []
    for (x, y) in coords:
        if ((x + y) % 2) == 0:
            even_points.append((x, y))
        else:
            odd_points.append((x, y))
    
    # Function to compute the sum of "max(|dx|, |dy|)" over all i<j
    # for a given list of points all having the same parity of (x+y).
    # From the derivation:
    #   dist = max(|x1 - x2|, |y1 - y2|) = (| (x1+y1) - (x2+y2) | + | (x1-y1) - (x2-y2) |) / 2
    #
    # So the sum of distances for all pairs = 1/2 * [ sum of pairwise |T1_i - T1_j| 
    #                                             + sum of pairwise |T2_i - T2_j| ],
    # where T1_i = x_i + y_i, T2_i = x_i - y_i.
    #
    # We can sum pairwise absolute differences in O(m log m) using a known formula:
    #   sum_{0 <= i < j < m} |A[i] - A[j]| = sum_{k=0 to m-1} A[k] * (2k - m + 1),
    # provided A is sorted non-decreasing and we use 0-based k.
    
    def sum_of_pairwise_abs(vals):
        # Assumes vals is sorted
        m = len(vals)
        s = 0
        for k, val in enumerate(vals):
            s += val * (2*k - m + 1)
        return s
    
    def sum_of_max_dist(points):
        if not points:
            return 0
        # Build T1 and T2
        T1 = [x+y for (x, y) in points]
        T2 = [x-y for (x, y) in points]
        T1.sort()
        T2.sort()
        
        sum_abs_t1 = sum_of_pairwise_abs(T1)
        sum_abs_t2 = sum_of_pairwise_abs(T2)
        # Each max-distance is (|T1 diff| + |T2 diff|)/2, so total is half the sum.
        return (sum_abs_t1 + sum_abs_t2) // 2
    
    ans = sum_of_max_dist(even_points) + sum_of_max_dist(odd_points)
    print(ans)

# Do not forget to call main()
main()