# YOUR CODE HERE
import sys

def solve():
    # Read N, L, R from the first line of standard input.
    # N is the length of the sequence A.
    # L and R define the closed interval [L, R].
    # Input values are integers, separated by spaces.
    n, l, r = map(int, sys.stdin.readline().split())
    
    # Read the integer sequence A = (A_1, A_2, ..., A_N) from the second line of standard input.
    # Input values are integers, separated by spaces.
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize an empty list to store the results X_i.
    results = []

    # Iterate through each element A_i in the sequence A.
    # In the loop, A_i is referred to as `val`.
    for val in a:
        # For each `val` (A_i), we need to find an integer X_i such that:
        # 1. L <= X_i <= R (X_i must be within the interval [L, R]).
        # 2. |X_i - val| is minimized among all integers Y satisfying L <= Y <= R.
        #
        # This problem asks for the element in the interval [L, R] that is closest to `val`.
        # This can be found by "clamping" `val` to the range [L, R]:
        # - If `val` is already within the interval [L, R] (i.e., L <= val <= R), then the closest element is `val` itself.
        # - If `val` is less than L (i.e., val < L), the closest element in [L, R] is the left endpoint, L.
        # - If `val` is greater than R (i.e., val > R), the closest element in [L, R] is the right endpoint, R.
        #
        # This clamping logic can be expressed concisely using `min` and `max` functions:
        # `min(val, r)` ensures the value does not exceed R.
        # `max(l, ...)` ensures the value is not less than L.
        # So, `max(l, min(val, r))` effectively clamps `val` into the range [L, R].
        
        x_i = max(l, min(val, r))
        
        # Append the calculated X_i to the results list.
        results.append(x_i)

    # Alternatively, the results can be computed more compactly using a list comprehension:
    # results = [max(l, min(val, r)) for val in a]
    
    # Print the resulting sequence X_1, X_2, ..., X_N to standard output.
    # The elements should be separated by spaces.
    # The `*` operator unpacks the `results` list into individual arguments for the `print` function.
    # By default, `print` separates its arguments with a single space and adds a newline at the end.
    print(*results)

# Call the solve function to execute the main logic of the program when the script is run.
solve()