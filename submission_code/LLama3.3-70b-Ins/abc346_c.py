def find_missing_sum():
    # Read input from stdin
    N, K = map(int, input().split())
    A = list(map(int, input().split()))

    # Create a set of unique numbers in the sequence A
    A_set = set(A)

    # Initialize the sum of missing numbers
    missing_sum = 0

    # Iterate over the range from 1 to K
    for i in range(1, K + 1):
        # Check if the number is not in the set A
        if i not in A_set:
            # Add the number to the sum of missing numbers
            missing_sum += i

    # Print the sum of missing numbers
    print(missing_sum)

# Call the function
find_missing_sum()