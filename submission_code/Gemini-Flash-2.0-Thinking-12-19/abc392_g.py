import sys

# Function to solve the problem
def solve():
    # Read the number of elements N
    N = int(sys.stdin.readline())

    # Read the elements of the set S
    # Split the line into integers and store them in a list
    S_list = list(map(int, sys.stdin.readline().split()))

    # Sort the list of elements.
    # Sorting is essential for the two-pointer approach to efficiently find pairs (A, C)
    # for a fixed middle element B.
    # The problem guarantees distinct elements, so sorting directly on the list is appropriate.
    sorted_S = sorted(S_list)

    # Initialize the count of fine triplets
    count = 0

    # Get the number of elements in the sorted list
    n = len(sorted_S)

    # Iterate through each element in the sorted list to consider it as the potential middle element B.
    # A fine triplet (A, B, C) requires A < B < C.
    # In the sorted list, if sorted_S[j] is B, then A must be some sorted_S[left] with left < j,
    # and C must be some sorted_S[right] with right > j.
    # This means that B cannot be the very first element (at index 0) or the very last element (at index n-1)
    # in the sorted list, as there would not be elements on both sides.
    # So, the index j for the potential middle element B ranges from 1 to n-2 (inclusive).
    for j in range(1, n - 1):
        # sorted_S[j] is our potential middle element B for the triplet (A, B, C).
        # The condition B - A = C - B is equivalent to A + C = 2 * B.
        # So, for the current sorted_S[j] (our potential B), we need to find pairs (A, C)
        # from the remaining elements in sorted_S such that A + C = 2 * sorted_S[j].
        target_sum = 2 * sorted_S[j]

        # Use two pointers, one starting from the beginning of the sorted list (index 0)
        # and the other starting from the end of the sorted list (index n-1).
        left = 0  # Pointer for potential A (sorted_S[left])
        right = n - 1 # Pointer for potential C (sorted_S[right])

        # The two pointers will move towards the middle.
        # The loop continues as long as the left pointer is strictly before the potential middle element (index j)
        # and the right pointer is strictly after the potential middle element (index j).
        # This ensures that we are considering elements for A that are strictly less than B (sorted_S[left] < sorted_S[j])
        # and elements for C that are strictly greater than B (sorted_S[right] > sorted_S[j]).
        while left < j and right > j:
            # Calculate the sum of the elements pointed to by the left and right pointers.
            current_sum = sorted_S[left] + sorted_S[right]

            if current_sum == target_sum:
                # If the current sum equals the target sum (2 * sorted_S[j]),
                # we have found a triplet (sorted_S[left], sorted_S[j], sorted_S[right])
                # that satisfies the conditions sorted_S[left] < sorted_S[j] < sorted_S[right] and
                # sorted_S[left] + sorted_S[right] = 2 * sorted_S[j].
                # This is a fine triplet.
                count += 1

                # Now we need to find other potential pairs (A, C) for the *same* middle element sorted_S[j].
                # If we move the left pointer to the right, sorted_S[left] increases, and the current_sum increases.
                # If we move the right pointer to the left, sorted_S[right] decreases, and the current_sum decreases.
                # Since we found a matching pair (sorted_S[left], sorted_S[right]), we move both pointers inwards
                # simultaneously to search for different pairs that might also sum to the target.
                left += 1
                right -= 1

            elif current_sum < target_sum:
                # If the current sum is less than the target sum, we need a larger sum.
                # The elements in sorted_S are in increasing order.
                # To increase the sum (current_sum = sorted_S[left] + sorted_S[right]), we can either
                # increase sorted_S[left] (by moving `left` to the right) or increase sorted_S[right] (by moving `right` to the right).
                # Since sorted_S[right] is already the largest element available from the right partition (right > j),
                # the effective way to increase the sum towards the target is to move the left pointer
                # to a larger element, i.e., move `left` to the right.
                left += 1

            else: # current_sum > target_sum
                # If the current sum is greater than the target sum, we need a smaller sum.
                # To decrease the sum (current_sum = sorted_S[left] + sorted_S[right]), we can either
                # decrease sorted_S[left] (by moving `left` to the left) or decrease sorted_S[right] (by moving `right` to the left).
                # Since sorted_S[left] is already the smallest element available from the left partition (left < j),
                # the effective way to decrease the sum towards the target is to move the right pointer
                # to a smaller element, i.e., move `right` to the left.
                right -= 1

    # After iterating through all possible middle elements and finding corresponding pairs,
    # the total count holds the number of fine triplets.
    # Print the final count.
    print(count)

# Call the solve function to execute the program
solve()