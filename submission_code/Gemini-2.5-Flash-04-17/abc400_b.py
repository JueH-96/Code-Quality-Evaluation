# YOUR CODE HERE
import sys

# Read the input integers N and M from a single line of standard input.
# The input line is expected to contain two space-separated integers.
line = sys.stdin.readline().split()
N = int(line[0])
M = int(line[1])

# Define the upper bound for the sum X as specified in the problem statement (10^9).
limit = 10**9

# Initialize the variable to store the running total sum of the series X = N^0 + N^1 + ... + N^M.
# The sum starts at 0.
total_sum = 0

# Initialize the variable to hold the value of the current term N^i that is being added in each iteration.
# The series starts with i=0, so the first term is N^0, which is 1 for any N >= 1.
current_term = 1

# Iterate through the exponents from i = 0 up to M (inclusive).
# This loop runs M + 1 times, processing each term from N^0 to N^M.
for i in range(M + 1):
    # Add the value of the current term (which is N^i in this iteration) to the running total sum.
    total_sum += current_term

    # After adding the current term, check if the accumulated total sum exceeds the defined limit (10^9).
    # Since the problem specifies N >= 1, all terms N^i are non-negative. If the partial sum
    # calculated so far (sum from N^0 up to N^i) is already greater than 10^9, then the final sum
    # for the entire series (up to N^M, where M >= i) will also be greater than 10^9,
    # because we would only be adding subsequent non-negative terms.
    if total_sum > limit:
        # If the sum exceeds the limit, print "inf" as required by the problem statement.
        # Then, terminate the program immediately using sys.exit().
        print("inf")
        sys.exit()

    # If there are more terms in the series to be calculated and added (i.e., if the current index i is less than M),
    # calculate the value of the next term in the series, N^(i+1). This is done by multiplying the current term (N^i) by N.
    # Python's integer type has arbitrary precision (its size is limited only by available memory).
    # Therefore, the multiplication `current_term * N` is safe and will not cause an integer overflow,
    # even if the intermediate `current_term` becomes very large before the total_sum exceeds the limit.
    if i < M:
        current_term *= N

# If the loop completes without the total_sum ever exceeding the limit (10^9), it means that
# the final calculated sum X = sum(N^i for i from 0 to M) is less than or equal to 10^9.
# In this case, print the final calculated sum.
print(total_sum)