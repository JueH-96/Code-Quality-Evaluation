import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    
    # Read array A
    A = list(map(int, sys.stdin.readline().split()))

    # The maximum possible value for A_i as per constraints.
    # This determines the size of our auxiliary arrays.
    MAX_A_VAL = 10**6 
    
    # sums_at_val[x] will store the sum of all occurrences of value x in A.
    # For example, if A contains two 5s, sums_at_val[5] will be 10.
    sums_at_val = [0] * (MAX_A_VAL + 1)
    
    # total_sum_A will store the sum of all elements in the input array A.
    total_sum_A = 0

    # First pass: Iterate through A to populate sums_at_val and calculate total_sum_A.
    for x in A:
        sums_at_val[x] += x
        total_sum_A += x
    
    # prefix_sums_up_to_val[x] will store the sum of all elements in A that are <= x.
    prefix_sums_up_to_val = [0] * (MAX_A_VAL + 1)
    
    current_prefix_sum = 0
    # Second pass: Compute prefix sums for sums_at_val.
    # This allows us to quickly query the sum of elements up to a certain value.
    for i in range(1, MAX_A_VAL + 1):
        current_prefix_sum += sums_at_val[i]
        prefix_sums_up_to_val[i] = current_prefix_sum
    
    results = []
    # Third pass: For each A_i in the original array, calculate the required sum.
    for val_A_i in A:
        # The sum of elements strictly greater than val_A_i is:
        # (Total sum of all elements) - (Sum of elements less than or equal to val_A_i)
        sum_greater_than_A_i = total_sum_A - prefix_sums_up_to_val[val_A_i]
        results.append(sum_greater_than_A_i)
    
    # Print the results, space-separated.
    sys.stdout.write(" ".join(map(str, results)) + "
")

# Call the solve function to execute the program.
solve()