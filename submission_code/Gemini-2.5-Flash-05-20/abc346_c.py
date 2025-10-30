import sys

def solve():
    # Read N and K from the first line
    N, K = map(int, sys.stdin.readline().split())

    # Read the sequence A from the second line
    A = list(map(int, sys.stdin.readline().split()))

    # Step 1: Calculate the sum of all integers from 1 to K.
    # The formula for the sum of an arithmetic series is n * (n + 1) / 2.
    # We use integer division // as the sum will always be an integer.
    total_sum_up_to_K = K * (K + 1) // 2

    # Step 2: Identify unique elements from A that are within the range [1, K].
    # Using a set handles duplicates in A automatically and provides fast lookups/additions.
    A_elements_to_subtract = set()
    for x in A:
        # Only consider elements of A that are within the specified range [1, K]
        if 1 <= x <= K:
            A_elements_to_subtract.add(x)

    # Step 3: Calculate the sum of these identified unique elements from A.
    # The sum() function works directly on a set.
    sum_of_A_elements_in_range = sum(A_elements_to_subtract)

    # Step 4: The final answer is the total sum from 1 to K minus the sum of
    # the elements from A that are within that range.
    result = total_sum_up_to_K - sum_of_A_elements_in_range

    # Print the result to standard output
    print(result)

# Call the solve function to run the program
solve()