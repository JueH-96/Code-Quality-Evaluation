# YOUR CODE HERE
import sys

# Read N and K
# N: length of sequence A (1 <= N <= 2e5)
# K: upper bound for integers to consider (1 <= K <= 2e9)
n, k = map(int, sys.stdin.readline().split())

# Calculate the sum of all integers from 1 to K
# The formula for the sum of integers from 1 to K is K * (K + 1) / 2.
# Use integer division // to ensure the result is an integer.
# K can be up to 2 * 10^9, so K * (K + 1) can be up to ~4 * 10^18.
# Python's arbitrary-precision integers handle this large value correctly.
total_sum_1_to_k = k * (k + 1) // 2

# Read the sequence A
# A_i: elements of the sequence (1 <= A_i <= 2e9)
# Reading all N elements into a list is efficient enough given N <= 2e5.
a_list = list(map(int, sys.stdin.readline().split()))

# Use a set to efficiently store and check for the presence of unique numbers from A.
# We only care about numbers from A that are within the range [1, K] because
# numbers outside this range do not affect which integers between 1 and K are missing.
present_numbers_in_range = set()

# Calculate the sum of unique numbers from A that are within the range [1, K].
# These are the numbers that we need to subtract from the total sum (1 to K).
sum_of_present_numbers_in_range = 0

# Iterate through the elements of the input sequence A.
for num in a_list:
    # Check if the current number `num` is within the relevant range [1, K].
    # The constraint 1 <= A_i means num is always >= 1, so we only strictly need num <= k,
    # but checking 1 <= num <= k is clearer regarding the range.
    if 1 <= num <= k:
        # Check if this number has already been added to our set of unique present numbers.
        if num not in present_numbers_in_range:
            # If it's a unique number within the range [1, K]:
            # Add it to the set so we don't count it again (handling duplicates in A).
            present_numbers_in_range.add(num)
            # Add its value to the sum of numbers that are present in the range [1, K].
            sum_of_present_numbers_in_range += num

# The sum of integers between 1 and K that do not appear in A
# is equal to the total sum of all integers from 1 to K
# minus the sum of the unique integers from A that are present in the range [1, K].
sum_of_missing_numbers = total_sum_1_to_k - sum_of_present_numbers_in_range

# Print the final result.
print(sum_of_missing_numbers)