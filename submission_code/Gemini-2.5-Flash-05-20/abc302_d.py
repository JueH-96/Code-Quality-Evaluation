import sys

def solve():
    # Read N, M, D from the first line of standard input.
    # N: number of gift candidates for Aoki
    # M: number of gift candidates for Snuke
    # D: maximum allowed difference in value between the two chosen gifts
    N, M, D = map(int, sys.stdin.readline().split())

    # Read the N gift values for Aoki from the second line.
    A = list(map(int, sys.stdin.readline().split()))

    # Read the M gift values for Snuke from the third line.
    B = list(map(int, sys.stdin.readline().split()))

    # Sort both lists of gift values in ascending order.
    # Sorting is crucial for the two-pointer approach to work efficiently.
    A.sort()
    B.sort()

    # Initialize max_sum to -1. This value will be printed if no valid pair of gifts
    # (i.e., a pair satisfying the difference condition) is found.
    max_sum = -1

    # Initialize two pointers, one for each sorted list.
    # We start the pointers at the end of the lists (pointing to the largest values).
    # This is because we want to maximize the sum (A_i + B_j), so it's most efficient
    # to consider larger values first.
    ptr_a = N - 1
    ptr_b = M - 1

    # Iterate while both pointers are within the valid bounds of their respective lists.
    while ptr_a >= 0 and ptr_b >= 0:
        current_a = A[ptr_a]
        current_b = B[ptr_b]

        # Calculate the absolute difference between the current gift values.
        diff = abs(current_a - current_b)

        # Case 1: The absolute difference is less than or equal to D.
        # This means the current pair (current_a, current_b) satisfies the condition.
        if diff <= D:
            # Update max_sum with the maximum of its current value and the sum of this valid pair.
            max_sum = max(max_sum, current_a + current_b)
            
            # Since we have found a valid pair involving current_a and current_b,
            # and we are trying to maximize the sum, any pair formed by current_a with a
            # smaller B value (B[k] where k < ptr_b) or current_b with a smaller A value
            # (A[k] where k < ptr_a) would result in a sum that is equal to or smaller
            # than current_a + current_b. Therefore, we can safely move both pointers
            # to consider the next smaller values.
            ptr_a -= 1
            ptr_b -= 1
        # Case 2: Current A value is significantly larger than current B value.
        # This means current_a - current_b > D.
        # To potentially satisfy the condition, we need to reduce the value from list A,
        # or increase the value from list B. Since we are already considering the largest
        # available B (current_b) for current_a, moving ptr_b to a smaller value would
        # only make the difference larger. Thus, we must move to a smaller A value.
        elif current_a > current_b:
            ptr_a -= 1
        # Case 3: Current B value is significantly larger than current A value.
        # This means current_b - current_a > D.
        # Similar to Case 2, to potentially satisfy the condition, we need to reduce the
        # value from list B. Moving ptr_a to a smaller value would only make the difference
        # larger. Thus, we must move to a smaller B value.
        else: # current_b > current_a
            ptr_b -= 1

    # After the loop finishes (meaning one or both pointers have gone out of bounds),
    # max_sum will hold the maximum sum found among all valid pairs, or -1 if no
    # such pair was found.
    print(max_sum)

# Call the solve function to execute the program.
if __name__ == "__main__":
    solve()