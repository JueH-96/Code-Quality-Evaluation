# YOUR CODE HERE
import sys

def solve():
    # Read the integer N, representing the number of distinct integers (1 to N).
    # N determines the range of numbers and the size multiplier for the sequence A.
    n = int(sys.stdin.readline())

    # Read the sequence A, which contains 3N integers.
    # The sequence is provided as space-separated integers on a single line.
    # Each integer from 1 to N appears exactly three times in A.
    a = list(map(int, sys.stdin.readline().split()))

    # Use a dictionary to keep track of the number of times each integer
    # (from 1 to N) has been encountered so far while iterating through A.
    # Key: integer `num` (from 1 to N)
    # Value: the count of occurrences of `num` seen up to the current position in A.
    counts = {}

    # Use a list to store pairs of `(middle_occurrence_index, number)`.
    # For each number `i` (from 1 to N), we will find the index `f(i)` of its
    # middle (second) occurrence in A. We will then store the pair `(f(i), i)`.
    # This list will eventually hold N such pairs, one for each unique number.
    middle_occurrences = []

    # Iterate through the input sequence A using a 0-based index `j`.
    # The loop runs from j = 0 to 3*N - 1, covering all elements of A.
    for j in range(3 * n):
        # Get the number at the current position `j` in the sequence A.
        num = a[j]

        # Retrieve the current count of `num` from the `counts` dictionary.
        # If `num` has not been seen before (`num` is not a key in `counts`),
        # `counts.get(num, 0)` returns the default value 0.
        current_count = counts.get(num, 0)
        # Increment the count because we have just encountered `num` again.
        current_count += 1
        # Update the count for `num` in the `counts` dictionary.
        counts[num] = current_count

        # The problem defines `f(i)` as the index of the *middle* occurrence of `i` in A.
        # Since each number appears exactly three times, the middle occurrence
        # is precisely the second time we encounter the number.
        # Check if the `current_count` for `num` has just become 2.
        if current_count == 2:
            # If the count is 2, this signifies that we have found the middle occurrence of `num`.
            # The problem statement uses 1-based indexing for positions in the sequence A.
            # The 1-based index corresponding to the current 0-based loop index `j` is `j + 1`.
            # This index `j + 1` is the middle occurrence index `f(num)`.
            # We store the pair `(f(num), num)`, which is `(j + 1, num)`, in our list.
            middle_occurrences.append((j + 1, num))

            # Note: We don't need to do anything specific when the count becomes 1 or 3
            # for the purpose of finding the middle index and the corresponding number.

    # After iterating through the entire sequence A, the `middle_occurrences` list
    # contains N pairs. Each pair corresponds to one unique number from 1 to N,
    # and is of the form `(f(i), i)`. However, the list is currently ordered by
    # the position of the middle occurrences as they were found, not necessarily sorted by the index `f(i)`.

    # We need to sort these pairs based on the middle occurrence index `f(i)`,
    # which is the first element of each tuple, in ascending order.
    # The `sort()` method sorts the list in place. `key=lambda pair: pair[0]` specifies
    # that the sorting should be based on the first element of each tuple (`pair[0]`).
    middle_occurrences.sort(key=lambda pair: pair[0])

    # Now, `middle_occurrences` is sorted according to the values of `f(i)`.
    # The sorted list has the structure `[(f(p1), p1), (f(p2), p2), ..., (f(pN), pN)]`
    # where `f(p1) <= f(p2) <= ... <= f(pN)`. The sequence `p1, p2, ..., pN`
    # is the desired permutation of the numbers 1 to N, sorted according to their `f(i)` values.

    # We need to extract this sequence of numbers `p1, p2, ..., pN`.
    # These numbers are the second elements (`item[1]`) of the tuples in the sorted list.
    # A list comprehension is used here for a concise way to create the new list.
    sorted_numbers = [item[1] for item in middle_occurrences]

    # Print the resulting sequence of numbers `p1, p2, ..., pN`.
    # The `*` operator unpacks the `sorted_numbers` list into individual arguments
    # for the `print` function. By default, `print` separates these arguments with spaces
    # and adds a newline character at the end. This matches the required output format.
    print(*sorted_numbers)

# Execute the `solve` function to run the main logic of the program when the script is executed.
solve()