import sys

def solve():
    # Read N, L, R from the first line of input.
    # sys.stdin.readline() is generally faster for competitive programming than input()
    # when dealing with large inputs.
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    L = int(line1[1])
    R = int(line1[2])

    # Read the sequence A from the second line of input.
    line2 = sys.stdin.readline().split()
    A = [int(x) for x in line2]

    results = []
    # Iterate through each element A_i in the sequence A.
    for a_val in A:
        # The problem asks to find X_i such that L <= X_i <= R
        # and |X_i - A_i| is minimized.
        # This is a classic "clamp" operation.
        #
        # If A_i is less than L (A_i < L), the closest value in [L, R] is L.
        #   Example: A_i=3, L=4, R=7 -> X_i=4.
        #
        # If A_i is greater than R (A_i > R), the closest value in [L, R] is R.
        #   Example: A_i=9, L=4, R=7 -> X_i=7.
        #
        # If A_i is within the range [L, R] (L <= A_i <= R), A_i itself is the closest value.
        #   Example: A_i=4, L=4, R=7 -> X_i=4.
        #   Example: A_i=7, L=4, R=7 -> X_i=7.
        #
        # This logic can be concisely expressed as: X_i = max(L, min(A_i, R)).
        #
        # Step 1: Ensure the value is not greater than R.
        # If a_val > R, it becomes R; otherwise, it remains a_val.
        temp_X_i = min(a_val, R)
        
        # Step 2: Ensure the value is not less than L.
        # If temp_X_i < L, it becomes L; otherwise, it remains temp_X_i.
        X_i = max(L, temp_X_i)
        
        results.append(str(X_i))
    
    # Print the results, separated by spaces.
    # sys.stdout.write() is used for potentially faster output.
    # A newline character is added at the end as required for standard output.
    sys.stdout.write(" ".join(results) + "
")

# Call the solve function to execute the program.
solve()