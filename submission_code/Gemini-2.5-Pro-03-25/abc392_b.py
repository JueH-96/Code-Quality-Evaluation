import sys

# Read N and M from the first line of standard input
# N is the upper bound of the range [1, N]
# M is the number of elements in the sequence A
n, m = map(int, sys.stdin.readline().split())

# Read the sequence A of M integers from the second line of standard input
# Constraints guarantee 1 <= M <= N, so this line always exists and has M elements.
# Each element A_i satisfies 1 <= A_i <= N and all A_i are distinct.
a = list(map(int, sys.stdin.readline().split()))

# Convert the list A into a set for efficient membership checking (O(1) average time complexity).
# This allows us to quickly determine if a number within the range [1, N] is present in the input sequence A.
set_a = set(a)

# Initialize an empty list to store the integers in the range [1, N] that are *not* present in A.
missing_numbers = []

# Iterate through all integers from 1 up to N (inclusive).
for i in range(1, n + 1):
    # Check if the current integer 'i' is present in the set `set_a`.
    # The `not in` operator performs the membership test efficiently.
    if i not in set_a:
        # If 'i' is not found in `set_a`, it means it's missing from the sequence A.
        # Add this missing number to our list `missing_numbers`.
        # Because we iterate `i` in increasing order (1, 2, ..., N),
        # the `missing_numbers` list will automatically be populated with the missing numbers
        # already sorted in ascending order.
        missing_numbers.append(i)

# Calculate the count of missing numbers, denoted as C in the problem description.
# This is simply the number of elements currently in the `missing_numbers` list.
count_missing = len(missing_numbers)

# Print the count C to standard output, followed by a newline character. This is the first line of the output.
print(count_missing)

# Print the sequence of missing numbers (X_1, X_2, ..., X_C) to standard output.
# The numbers should be separated by spaces. This forms the second line of the output.
# The `*` operator before `missing_numbers` unpacks the list elements.
# `print(*missing_numbers)` is equivalent to `print(missing_numbers[0], missing_numbers[1], ..., missing_numbers[C-1])`.
# The `print` function, by default, separates arguments with spaces.
# If `missing_numbers` is empty (i.e., count_missing is 0), `print(*[])` correctly
# prints nothing followed by a newline. This ensures that for C=0 cases, the second line of output is empty,
# as required by the problem specification (e.g., Sample Output 2).
print(*missing_numbers)