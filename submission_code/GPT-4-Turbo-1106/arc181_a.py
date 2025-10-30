def min_operations_to_sort(P):
    N = len(P)
    left = 0
    right = N - 1

    # Find the leftmost element that is not in the correct position
    while left < N and P[left] == left + 1:
        left += 1

    # Find the rightmost element that is not in the correct position
    while right >= 0 and P[right] == right + 1:
        right -= 1

    # If the entire array is already sorted
    if left > right:
        return 0

    # Check if the elements between left and right are all out of place
    for i in range(left, right + 1):
        if P[i] == i + 1:
            return 2

    return 1

# Read the number of test cases
T = int(input().strip())

# Process each test case
for _ in range(T):
    # Read the length of the permutation and the permutation itself
    N = int(input().strip())
    P = list(map(int, input().strip().split()))

    # Calculate and print the minimum number of operations required
    print(min_operations_to_sort(P))