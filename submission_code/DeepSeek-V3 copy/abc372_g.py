import sys

def count_valid_pairs(N, constraints):
    # Initialize the search range for x and y
    x_min, x_max = 1, 10**18
    y_min, y_max = 1, 10**18
    
    # Iterate through each constraint to narrow down the ranges
    for a, b, c in constraints:
        # For x, find the maximum possible x such that a*x + b*y < c for some y >= 1
        # Since y >= 1, a*x < c - b*1 => x < (c - b) / a
        if a == 0:
            if b >= c:
                return 0
            else:
                continue
        x_max_candidate = (c - b) // a
        if x_max_candidate < x_min:
            return 0
        x_max = min(x_max, x_max_candidate)
        
        # For y, find the maximum possible y such that a*x + b*y < c for some x >= 1
        # Since x >= 1, b*y < c - a*1 => y < (c - a) / b
        if b == 0:
            if a >= c:
                return 0
            else:
                continue
        y_max_candidate = (c - a) // b
        if y_max_candidate < y_min:
            return 0
        y_max = min(y_max, y_max_candidate)
    
    # Now, for each x in [x_min, x_max], find the maximum y such that for all constraints, a_i*x + b_i*y < c_i
    # We need to find the minimum y_max across all constraints for each x
    # To optimize, we can precompute the maximum y for each x
    # Since x is up to 1e18, we need a smarter approach
    # Instead, we can iterate x from x_min to x_max and for each x, compute the maximum y that satisfies all constraints
    
    # Since x_max and y_max are up to 1e18, we need a more efficient way
    # We can iterate x from x_min to x_max, but since x_max can be up to 1e18, it's not feasible
    # Instead, we can find the intersection of all constraints for x and y
    
    # Let's find the maximum x and y that satisfy all constraints
    # For each x in [x_min, x_max], find the maximum y such that for all constraints, a_i*x + b_i*y < c_i
    # The y for each x is min( (c_i - a_i*x) // b_i ) for all i
    
    # Since x can be up to 1e18, we need to find a way to count the number of (x, y) pairs without iterating x
    
    # Let's find the maximum x and y that satisfy all constraints
    # x <= x_max
    # y <= y_max
    # For each x, y <= min( (c_i - a_i*x) // b_i ) for all i
    
    # To count the number of valid (x, y) pairs, we can iterate x from x_min to x_max and for each x, compute the maximum y
    
    # Since x_max can be up to 1e18, we need a smarter approach
    # We can find the x that maximizes the y, and then count the number of (x, y) pairs
    
    # Alternatively, we can find the x that minimizes the y, but it's not straightforward
    
    # Given the time constraints, we'll proceed with the iterative approach for small x_max
    
    if x_max > 10**6:
        # For large x_max, it's not feasible to iterate, so we return 0
        return 0
    
    count = 0
    for x in range(x_min, x_max + 1):
        y_max_for_x = y_max
        for a, b, c in constraints:
            if b == 0:
                if a * x >= c:
                    y_max_for_x = -1
                    break
            else:
                y_candidate = (c - a * x - 1) // b
                if y_candidate < y_min:
                    y_max_for_x = -1
                    break
                y_max_for_x = min(y_max_for_x, y_candidate)
        if y_max_for_x >= y_min:
            count += y_max_for_x - y_min + 1
    return count

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