import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    total_sum = 0
    # last_occurrence stores the 0-indexed position of the last time each value was encountered.
    last_occurrence = {} 

    for i in range(N):
        current_value = A[i]
        
        # Get the previous occurrence index of current_value.
        # If this value has not been seen before, .get() returns the default value -1.
        prev_idx = last_occurrence.get(current_value, -1)
        
        # Calculate the number of valid starting points (l-1) for a subarray
        # such that A[i] is the first occurrence of current_value in A[l-1 ... i].
        # This means l-1 must be strictly greater than prev_idx, and less than or equal to i.
        # So, l-1 ranges from (prev_idx + 1) to i.
        # The number of such choices for l-1 is i - (prev_idx + 1) + 1 = i - prev_idx.
        num_left_choices = i - prev_idx
        
        # Calculate the number of valid ending points (j-1) for a subarray
        # starting from or before i, and ending at or after i.
        # j-1 must be greater than or equal to i, and less than or equal to N-1.
        # So, j-1 ranges from i to (N-1).
        # The number of such choices for j-1 is (N-1) - i + 1 = N - i.
        num_right_choices = N - i
        
        # The contribution of A[i] to the total sum is the product of these choices.
        # This is because A[i] will be counted as a distinct element in every subarray
        # formed by combining one of these 'left' choices with one of these 'right' choices.
        total_sum += num_left_choices * num_right_choices
        
        # Update the last seen position for the current value to its current index.
        last_occurrence[current_value] = i
        
    print(total_sum)

# Call the solve function to execute the program
solve()