# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input sequence, performs the insertion procedure,
    and prints the final sequence.
    """
    # Read the length of the sequence
    n = int(sys.stdin.readline())
    # Read the sequence itself
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize the result list with the first element of the input sequence
    result = [a[0]]

    # Iterate through the adjacent pairs of the original sequence A
    # The loop runs from the first element up to the second-to-last element
    for i in range(n - 1):
        current = a[i]    # The current element A_i
        next_val = a[i+1] # The next element A_{i+1}

        # Calculate the absolute difference between the adjacent pair
        diff = abs(current - next_val)

        # Check if the absolute difference is exactly 1
        if diff == 1:
            # If the difference is 1, no insertion is needed between this pair.
            # Simply append the next element to the result.
            result.append(next_val)
        # Check if the current element is less than the next element
        elif current < next_val:
            # If current < next_val and the difference is greater than 1,
            # we need to insert the numbers from current + 1 up to next_val - 1.
            # The `range(start, stop)` function generates numbers from start up to stop - 1.
            # To include next_val in the sequence, the range should go up to next_val + 1.
            # The sequence to append is (current + 1, current + 2, ..., next_val).
            result.extend(range(current + 1, next_val + 1))
        # The remaining case is current > next_val
        else: # current > next_val and difference > 1
            # If current > next_val and the difference is greater than 1,
            # we need to insert the numbers from current - 1 down to next_val + 1.
            # The `range(start, stop, step)` function with step = -1 generates numbers in decreasing order.
            # The stop value is exclusive. To include next_val, the range should stop at next_val - 1.
            # The sequence to append is (current - 1, current - 2, ..., next_val).
            result.extend(range(current - 1, next_val - 1, -1))

    # Print the elements of the final result list, separated by spaces.
    # The * operator unpacks the list elements as arguments to print.
    print(*result)

# Call the solve function to execute the logic when the script runs
solve()