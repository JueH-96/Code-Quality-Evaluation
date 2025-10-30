import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MAX_A_VAL = 10**6 # Maximum possible value for A_i as per constraints

    # counts[v] stores the frequency of value v in A
    # Initialize with zeros up to MAX_A_VAL
    counts = [0] * (MAX_A_VAL + 1)
    for x in A:
        counts[x] += 1

    # prefix_counts[v] stores the sum of frequencies from 1 to v
    # This allows O(1) queries for sum of counts in a range.
    prefix_counts = [0] * (MAX_A_VAL + 1)
    for i in range(1, MAX_A_VAL + 1):
        prefix_counts[i] = prefix_counts[i-1] + counts[i]

    # Helper function to get the total count of numbers in A within a specified range [start, end].
    # This is equivalent to sum(counts[x] for x in range(start, end + 1)).
    def get_count_in_range(start, end):
        if start > end:
            return 0
        # prefix_counts[end] gives sum of counts up to 'end'
        # prefix_counts[start-1] gives sum of counts up to 'start-1'
        # Their difference is the sum of counts from 'start' to 'end'.
        return prefix_counts[end] - prefix_counts[start-1]

    total_sum = 0

    # Case 1: A_i == A_j (where i != j)
    # For each unique value 'v' present in the array A:
    # If 'v' appears 'counts[v]' times, the number of pairs (A_i, A_j) with A_i=A_j=v and i<j
    # is given by the combination formula C(counts[v], 2) = counts[v] * (counts[v]-1) / 2.
    # For these pairs, max(v,v)/min(v,v) = v/v = 1.
    for v in range(1, MAX_A_VAL + 1):
        if counts[v] >= 2:
            total_sum += (counts[v] * (counts[v] - 1) // 2) * 1

    # Case 2: A_i != A_j
    # Let y = min(A_i, A_j) and x = max(A_i, A_j). We must have y < x.
    # The term to add is floor(x/y).
    # We iterate through all possible values 'y' from 1 to MAX_A_VAL.
    for y in range(1, MAX_A_VAL + 1):
        if counts[y] == 0:
            continue # If value 'y' is not present in A, it cannot be min(A_i, A_j)

        # For a fixed 'y', we iterate through possible integer values of k = floor(x/y).
        # 'k' can range from 1 up to MAX_A_VAL // y.
        # The values of 'x' for which floor(x/y) = k are in the range [k*y, (k+1)*y - 1].
        # However, 'x' must also be strictly greater than 'y'.
        for k_val in range(1, MAX_A_VAL // y + 1):
            # Calculate the natural lower and upper bounds for x based on k_val and y
            lower_x_bound_natural = k_val * y
            upper_x_bound_natural = (k_val + 1) * y - 1

            # Adjust lower_x_bound to ensure x > y.
            # If k_val * y is less than or equal to y, it means k_val must be 1.
            # In this scenario, the smallest x we consider must be y+1.
            effective_lower_x_bound = max(y + 1, lower_x_bound_natural)
            
            # Ensure upper_x_bound does not exceed the maximum possible value in A.
            effective_upper_x_bound = min(upper_x_bound_natural, MAX_A_VAL)
            
            # Get the total frequency of numbers 'x' that fall into the current valid range
            # [effective_lower_x_bound, effective_upper_x_bound].
            num_x_in_range = get_count_in_range(effective_lower_x_bound, effective_upper_x_bound)
            
            if num_x_in_range > 0:
                # The contribution for this fixed 'y' and 'k_val' is:
                # counts[y] (number of choices for min(A_i,A_j)) * 
                # num_x_in_range (total number of choices for max(A_i,A_j) from relevant x values) *
                # k_val (the floor(x/y) value for these pairs).
                total_sum += counts[y] * num_x_in_range * k_val

    # Print the final calculated total sum.
    sys.stdout.write(str(total_sum) + '
')

# Call the solve function to execute the program.
solve()