import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.

    The method involves a coordinate transformation and an efficient algorithm
    for calculating the sum of pairwise distances in one dimension.
    """

    def calculate_1d_sum(arr):
        """
        Calculates the sum of absolute differences for all pairs in a 1D array.
        The sum is sum_{i<j} |arr[j] - arr[i]|.
        This is done in O(k log k) time, where k is the length of the array.
        The formula used after sorting is sum_{i=0}^{k-1} arr[i] * (2*i - k + 1).
        An equivalent iterative calculation is used for implementation.
        """
        k = len(arr)
        if k < 2:
            return 0
        
        arr.sort()
        
        total_sum = 0
        current_prefix_sum = 0
        # This loop calculates sum_{i=0..k-1} sum_{j=0..i-1} (arr[i] - arr[j])
        for i in range(k):
            # For each arr[i], it contributes (arr[i] - arr[j]) for all j < i.
            # The sum of these contributions is i * arr[i] - sum(arr[j] for j < i).
            total_sum += i * arr[i] - current_prefix_sum
            current_prefix_sum += arr[i]
            
        return total_sum

    # Read number of points
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
    except (IOError, ValueError):
        return

    points_even = []
    points_odd = []

    # Read points and partition them based on parity of X+Y
    for _ in range(N):
        try:
            line = sys.stdin.readline()
            if not line: break
            X, Y = map(int, line.split())
        except (IOError, ValueError):
            break
        
        if (X + Y) % 2 == 0:
            points_even.append((X, Y))
        else:
            points_odd.append((X, Y))

    total_distance = 0

    # Process each group of points (same parity of X+Y)
    for group in [points_even, points_odd]:
        if len(group) < 2:
            continue
        
        # Transform coordinates: (X, Y) -> (U, V) = (X+Y, X-Y)
        # The distance formula simplifies in this new coordinate system.
        # dist((X1,Y1), (X2,Y2)) = max(|X2-X1|, |Y2-Y1|)
        # which is equivalent to 1/2 * (|U2-U1| + |V2-V1|), i.e.,
        # half the Manhattan distance in the (U,V) space.
        
        # We need to compute sum_{i<j} 1/2 * (|U_j-U_i| + |V_j-V_i|)
        # = 1/2 * ( sum_{i<j} |U_j-U_i| + sum_{i<j} |V_j-V_i| )
        
        u_coords = [p[0] + p[1] for p in group]
        v_coords = [p[0] - p[1] for p in group]
        
        # Calculate sum of pairwise distances for U and V coordinates separately
        sum_dist_u = calculate_1d_sum(u_coords)
        sum_dist_v = calculate_1d_sum(v_coords)
        
        # Add the contribution of this group to the total distance.
        # The sum is guaranteed to be even, so integer division is safe.
        total_distance += (sum_dist_u + sum_dist_v) // 2
        
    print(total_distance)

if __name__ == "__main__":
    solve()