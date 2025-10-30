import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Store S and D coordinates separately for points with even S and odd S
    points_even_S = [] 
    points_even_D = [] 
    points_odd_S = []  
    points_odd_D = []  

    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        S = X + Y
        D = X - Y
        
        # Group points based on the parity of their S (X+Y) coordinate.
        # This implies the same parity for D (X-Y) as well.
        if S % 2 == 0:
            points_even_S.append(S)
            points_even_D.append(D)
        else:
            points_odd_S.append(S)
            points_odd_D.append(D)

    total_sum_of_distances = 0

    # Helper function to calculate sum of |v_j - v_i| for all pairs (i < j) in a list
    def calculate_sum_of_absolute_differences(values):
        if not values:
            return 0
        
        values.sort() # Sort the coordinates in ascending order
        
        current_prefix_sum = 0
        sum_abs_diff = 0
        
        # The sum is Sum_{j=0 to M-1} (values[j] * j - Sum_{i=0 to j-1} values[i])
        for i in range(len(values)):
            sum_abs_diff += values[i] * i - current_prefix_sum
            current_prefix_sum += values[i]
            
        return sum_abs_diff

    # Calculate sums for S coordinates in the even group
    total_sum_of_distances += calculate_sum_of_absolute_differences(points_even_S)
    # Calculate sums for D coordinates in the even group
    total_sum_of_distances += calculate_sum_of_absolute_differences(points_even_D)
    
    # Calculate sums for S coordinates in the odd group
    total_sum_of_distances += calculate_sum_of_absolute_differences(points_odd_S)
    # Calculate sums for D coordinates in the odd group
    total_sum_of_distances += calculate_sum_of_absolute_differences(points_odd_D)

    # Each individual distance dist(P_i, P_j) was defined as (|S_j-S_i| + |D_j-D_i|) / 2.
    # We summed all |S_j-S_i| terms and all |D_j-D_i| terms separately.
    # Therefore, the final total sum needs to be divided by 2.
    print(total_sum_of_distances // 2)

solve()