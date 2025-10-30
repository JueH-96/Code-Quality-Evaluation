import sys

# Use sys.stdin.readline for faster input
input = sys.stdin.readline

def solve():
    N = int(input())
    points = []
    for _ in range(N):
        x, y = map(int, input().split())
        points.append((x, y))

    points_even_v = []
    points_odd_v = []

    # Group points based on the parity of X - Y
    for x, y in points:
        if (x - y) % 2 == 0:
            points_even_v.append((x, y))
        else:
            points_odd_v.append((x, y))

    total_distance_sum = 0

    def calculate_group_sum(point_list):
        m = len(point_list)
        if m < 2:
            return 0

        # Calculate u = X + Y and v = X - Y for each point
        u_values = [p[0] + p[1] for p in point_list]
        v_values = [p[0] - p[1] for p in point_list]

        # Sort the u and v values
        u_values.sort()
        v_values.sort()

        # The sum of distances is Sum_{0 <= i < j <= m-1} ( |u_j - u_i| / 2 + |v_j - v_i| / 2 )
        # = (1/2) * ( Sum_{0 <= i < j <= m-1} |u_j - u_i| + Sum_{0 <= i < j <= m-1} |v_j - v_i| )
        # For a sorted list z, Sum_{0 <= i < j <= m-1} |z_j - z_i| = Sum_{k=0}^{m-1} (2k - m + 1) * z[k]

        sum_U = 0
        for k in range(m):
            sum_U += (2 * k - m + 1) * u_values[k]

        sum_V = 0
        for k in range(m):
            sum_V += (2 * k - m + 1) * v_values[k]

        # sum_U and sum_V are integers and guaranteed to be even in this problem context.
        # Their sum is also even. Integer division // 2 is safe.
        return (sum_U + sum_V) // 2

    # Calculate the sum for each group and add to the total
    total_distance_sum += calculate_group_sum(points_even_v)
    total_distance_sum += calculate_group_sum(points_odd_v)

    print(total_distance_sum)

solve()