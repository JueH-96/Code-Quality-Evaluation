# YOUR CODE HERE
import sys

def solve():
    """
    Reads two positive integers N and M from standard input.
    Calculates the sum X = N^0 + N^1 + ... + N^M.
    Prints the value of X if X is less than or equal to 10^9.
    Prints "inf" if X is greater than 10^9.
    """
    try:
        # Read the input line and split it into two parts
        line = sys.stdin.readline().split()
        
        # Ensure exactly two inputs were provided
        if len(line) != 2:
             # In competitive programming contexts, we usually assume input format is correct.
             # If error handling were required, it would go here.
             # For this problem, assume valid input based on constraints.
             return

        # Convert inputs to integers
        n = int(line[0])
        m = int(line[1])

        # Problem constraints specify N >= 1, M >= 1.
        # We don't typically need to re-validate constraints in timed contests
        # unless the problem statement implies otherwise or it helps catch edge cases.

    except ValueError:
        # Handle cases where conversion to int fails (e.g., non-numeric input)
        # Again, typically assume valid input in competitive programming.
        # print("Error: Inputs must be integers.", file=sys.stderr)
        return
        
    # Define the limit
    limit = 10**9

    # --- Special Case: N = 1 ---
    # If N is 1, the sum simplifies significantly.
    if n == 1:
        # X = 1^0 + 1^1 + ... + 1^M = 1 + 1 + ... + 1 (M+1 times)
        result = m + 1
        # According to the constraints, M <= 100, so M+1 <= 101.
        # Since 101 is always less than or equal to 10^9, this sum never exceeds the limit.
        print(result)
        return

    # --- General Case: N > 1 ---
    current_sum = 0
    current_power = 1 # Represents N^i, starting with N^0 = 1

    # Loop through the powers from i = 0 to M
    for i in range(m + 1):
        # Check 1: Prevent overflow before adding the current power term (N^i).
        # Check if current_sum + current_power would exceed the limit.
        # Use the condition current_sum > limit - current_power to avoid potential
        # overflow during the check itself if current_power is very large.
        if current_sum > limit - current_power:
            print("inf")
            return # Exit if overflow detected
        
        # Add the current power term to the sum.
        # Python's arbitrary-precision integers handle large intermediate sums.
        current_sum += current_power

        # Check 2: Prepare for the next iteration (calculate N^(i+1)).
        # This step is only needed if we haven't processed the last term (i.e., i < M).
        if i < m:
            # Prevent overflow *before* calculating the next power term (current_power * N).
            # Check if the next power term (N^(i+1)) would exceed the limit.
            # If N * current_power > limit, the next term itself is too large.
            # A safe way to check this without potentially overflowing the multiplication
            # is to use integer division: current_power > limit // n.
            # This relies on N >= 1 (guaranteed by constraints).
            # If this condition is true, the next term N^(i+1) will be > limit.
            # Adding such a large term will inevitably cause the sum to exceed the limit.
            if current_power > limit // n:
                print("inf")
                return # Exit if overflow detected for the next term
            
            # Calculate the next power term N^(i+1) = N^i * N.
            # Python handles the large integer multiplication.
            current_power *= n

    # If the loop completes without returning "inf", it means all terms were added
    # without the sum ever exceeding the limit. The final sum is valid.
    print(current_sum)

# Execute the solve function when the script runs.
solve()
# YOUR CODE HERE