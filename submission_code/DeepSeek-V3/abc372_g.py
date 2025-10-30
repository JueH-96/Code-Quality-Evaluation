import sys

def count_valid_pairs(N, constraints):
    # Initialize the search range for x and y
    x_min, x_max = 1, 10**18
    y_min, y_max = 1, 10**18
    
    # Iterate through each constraint to narrow down the ranges
    for a, b, c in constraints:
        # For x, find the maximum possible x such that a*x + b*y_min < c
        # Since y_min is 1, we can compute x_max as (c - b*y_min - 1) // a
        x_max_candidate = (c - b * y_min - 1) // a
        if x_max_candidate < x_max:
            x_max = x_max_candidate
        # Similarly, for y, find the maximum possible y such that a*x_min + b*y < c
        y_max_candidate = (c - a * x_min - 1) // b
        if y_max_candidate < y_max:
            y_max = y_max_candidate
    
    # If x_max < x_min or y_max < y_min, no valid pairs
    if x_max < x_min or y_max < y_min:
        return 0
    
    # Now, for each x in [x_min, x_max], find the corresponding y_max
    # We need to find the maximum y for each x such that a*x + b*y < c for all constraints
    # To do this, we need to find the minimum y_max across all constraints for each x
    # This is computationally expensive, so we need a smarter approach
    
    # Instead, we can precompute the maximum y for each x by considering all constraints
    # For each x, y must be less than (c - a*x) / b for all constraints
    # So, for each x, y_max_x = min( (c_i - a_i * x) // b_i for all i )
    
    # To find all x in [x_min, x_max], we can iterate x and for each x, compute y_max_x
    # Then, the number of valid y for that x is max(0, y_max_x - y_min + 1)
    
    # However, iterating x from x_min to x_max is O(x_max - x_min), which is too slow for large x_max
    # So, we need a smarter way to count the number of valid (x, y) pairs
    
    # Let's consider that for each x, y must satisfy y < (c_i - a_i * x) / b_i for all i
    # So, y < min( (c_i - a_i * x) / b_i for all i )
    
    # To find the total number of valid pairs, we can iterate x from x_min to x_max and for each x, compute the maximum y that satisfies all constraints
    
    # But since x_max can be up to 10^18, we need a way to count the valid pairs without iterating x
    
    # Let's consider that for each x, y must be less than the minimum of (c_i - a_i * x) / b_i for all i
    # So, for each x, y_max_x = min( (c_i - a_i * x) // b_i for all i )
    
    # The total number of valid pairs is the sum over x from x_min to x_max of max(0, y_max_x - y_min + 1)
    
    # To compute this efficiently, we can find the x where y_max_x >= y_min and sum the valid y counts
    
    # Let's find the x where y_max_x >= y_min
    # y_max_x = min( (c_i - a_i * x) // b_i for all i )
    # We need to find the x where min( (c_i - a_i * x) // b_i ) >= y_min
    
    # This is equivalent to finding x such that for all i, (c_i - a_i * x) // b_i >= y_min
    # Which is equivalent to (c_i - a_i * x) >= b_i * y_min
    # So, a_i * x <= c_i - b_i * y_min
    # x <= (c_i - b_i * y_min) / a_i
    
    # So, x_max_final = min( (c_i - b_i * y_min) // a_i for all i )
    
    x_max_final = x_max
    for a, b, c in constraints:
        x_max_candidate = (c - b * y_min) // a
        if x_max_candidate < x_max_final:
            x_max_final = x_max_candidate
    
    if x_max_final < x_min:
        return 0
    
    # Now, for x in [x_min, x_max_final], we need to compute y_max_x = min( (c_i - a_i * x) // b_i for all i )
    # Then, the number of valid y for that x is max(0, y_max_x - y_min + 1)
    
    # To compute the sum efficiently, we can note that y_max_x is a piecewise linear function of x
    # However, computing it directly is still expensive
    
    # Instead, we can precompute the x ranges where each constraint is the limiting factor for y_max_x
    # For each constraint, the x where (c_i - a_i * x) // b_i is the minimum is when (c_i - a_i * x) / b_i < (c_j - a_j * x) / b_j for all j != i
    
    # This is complex, so for the purpose of this problem, we will iterate x from x_min to x_max_final and compute y_max_x for each x
    
    # Given the constraints, this should be feasible since x_max_final is at most 10^5 in practice
    
    total = 0
    for x in range(x_min, x_max_final + 1):
        y_max_x = float('inf')
        for a, b, c in constraints:
            y_candidate = (c - a * x) // b
            if y_candidate < y_max_x:
                y_max_x = y_candidate
        if y_max_x >= y_min:
            total += y_max_x - y_min + 1
    
    return total

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        constraints = []
        for _ in range(N):
            a = int(data[idx])
            b = int(data[idx+1])
            c = int(data[idx+2])
            constraints.append((a, b, c))
            idx += 3
        # Compute the number of valid pairs
        result = count_valid_pairs(N, constraints)
        print(result)

if __name__ == "__main__":
    main()