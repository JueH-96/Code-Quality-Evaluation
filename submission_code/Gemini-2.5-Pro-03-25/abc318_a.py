# YOUR CODE HERE
import sys

def solve():
    """
    Reads input N, M, P and calculates the number of full moon days
    between day 1 and day N (inclusive).
    """
    # Read input N, M, P from standard input
    # Input is expected to be on a single line separated by spaces.
    line = sys.stdin.readline().split()
    n = int(line[0]) # The last day to consider (inclusive)
    m = int(line[1]) # The first day of a full moon
    p = int(line[2]) # The period between full moons

    # Check if the first full moon day M is after the last day N.
    if m > n:
        # If the first full moon occurs after day N, then there are no
        # full moon days within the range [1, N].
        print(0)
    else:
        # If the first full moon day M is on or before day N,
        # then there is at least one full moon day (day M itself).
        # The full moon days form an arithmetic progression: M, M + P, M + 2P, ...
        # We need to find how many terms of this sequence are less than or equal to N.

        # Let a term be D = M + k * P, where k >= 0 is an integer.
        # The index k=0 corresponds to the first full moon day M.
        # The index k=1 corresponds to the second full moon day M + P, and so on.

        # We require the full moon day D to be within the observation period, i.e., D <= N.
        # So, M + k * P <= N.

        # Rearranging the inequality to solve for k:
        # k * P <= N - M

        # Since P is guaranteed to be positive (P >= 1 based on constraints 1 <= M <= P),
        # we can divide by P without changing the inequality direction:
        # k <= (N - M) / P

        # We need to count the number of non-negative integers k (k >= 0)
        # that satisfy this condition k <= (N - M) / P.
        # The possible integer values for k are 0, 1, 2, ..., up to the largest integer
        # less than or equal to (N - M) / P. This largest integer is floor((N - M) / P).

        # The number of integers in the sequence 0, 1, ..., floor_val is floor_val - 0 + 1 = floor_val + 1.
        
        # In Python, integer division `//` performs floor division.
        # So, floor((N - M) / P) is equivalent to (n - m) // p.
        # The number of full moon days is the number of possible values for k.
        count = (n - m) // p + 1

        # Print the final count to standard output.
        print(count)

# Call the solve function to execute the logic
solve()