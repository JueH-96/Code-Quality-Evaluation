def calculate_sum(X, Y):
    # Group points based on (x mod 2, y mod 2)
    groups = [[], [], [], []]
    for x, y in zip(X, Y):
        idx = (x % 2) * 2 + (y % 2)
        groups[idx].append((x, y))
    
    # Sort groups by x and y
    for idx in range(4):
        groups[idx].sort()
    
    total_sum = 0
    
    # Helper function to compute sum of (xj - xi + 1) // 2 for sorted list
    def sum_within_group(xs):
        n = len(xs)
        sorted_xs = sorted(xs)
        cum_sum = [0] * (n + 1)
        for i in range(n):
            cum_sum[i + 1] = cum_sum[i] + sorted_xs[i]
        sum_dist = 0
        for i in range(n):
            x_i = sorted_xs[i]
            # Sum (x_j - x_i + 1) // 2 for j > i
            count = n - i - 1
            sum_xj = cum_sum[-1] - cum_sum[i + 1]
            sum_dist += (sum_xj - count * x_i + count) // 2
        return sum_dist
    
    # Sum within same group for all four groups
    for group in groups:
        xs = [x for x, y in group]
        total_sum += sum_within_group(xs)
        total_sum += sum_within_group([y for x, y in group])
    
    # Sum between group A and D, and group B and C
    for group1, group2 in [(0, 3), (1, 2)]:
        xs1 = [x for x, y in groups[group1]]
        xs2 = [x for x, y in groups[group2]]
        ys1 = [y for x, y in groups[group1]]
        ys2 = [y for x, y in groups[group2]]
        
        xs1_sorted = sorted(xs1)
        xs2_sorted = sorted(xs2)
        ys1_sorted = sorted(ys1)
        ys2_sorted = sorted(ys2)
        
        # Sum (xj - xi + 1) // 2 for all pairs (xi in group1, xj in group2)
        m, n = len(xs1_sorted), len(xs2_sorted)
        i = 0
        sum_dist = 0
        for xj in xs2_sorted:
            while i < m and xs1_sorted[i] <= xj:
                xi = xs1_sorted[i]
                sum_dist += (xj - xi + 1) // 2
                i += 1
        total_sum += sum_dist
        
        # Similarly for y coordinates
        m, n = len(ys1_sorted), len(ys2_sorted)
        i = 0
        sum_dist = 0
        for yj in ys2_sorted:
            while i < m and ys1_sorted[i] <= yj:
                yi = ys1_sorted[i]
                sum_dist += (yj - yi + 1) // 2
                i += 1
        total_sum += sum_dist
    
    return total_sum

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = []
    Y = []
    for i in range(1, 2*N + 1, 2):
        X.append(int(data[i]))
        Y.append(int(data[i+1]))
    print(calculate_sum(X, Y))

if __name__ == "__main__":
    main()