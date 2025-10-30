import math
from collections import Counter
import sys

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())
    # Read S from standard input and remove trailing newline
    S = sys.stdin.readline().strip()

    # Pre-process S: count the frequency of each digit in S.
    # Counter({'4': 1, '3': 1, '2': 1, '0': 1}) for S="4320"
    s_digit_counts = Counter(S)

    # Use a set to store unique square numbers found.
    # This automatically handles the requirement "different permutations are not distinguished
    # if they result in the same number".
    found_squares = set()

    # Determine the maximum possible integer whose square we need to check.
    # The largest number that can be formed with N digits is 10^N - 1 (e.g., 99...9 for N times).
    # We need to find k such that k*k <= (10^N - 1).
    # The smallest k is 0 (0*0 = 0).
    # The `if N > 0 else 0` handles the theoretical case N=0, though constraints state N >= 1.
    max_k = int(math.sqrt(10**N - 1)) if N > 0 else 0

    # Iterate through potential square roots (k values)
    for k in range(max_k + 1):
        num = k * k  # Calculate the square number

        # Convert the number to a string.
        # Use zfill(N) to pad with leading zeros to ensure the string representation
        # has exactly N digits. This is crucial for matching digit counts correctly,
        # especially for numbers like 1 (when N=3, it should be "001").
        num_str_padded = str(num).zfill(N)

        # Count digit frequencies for the current N-digit number string.
        num_digit_counts = Counter(num_str_padded)

        # Compare the digit frequencies of the current square number with the
        # digit frequencies of the input string S.
        # If they match, it means 'num' can be formed by permuting the digits of S.
        if num_digit_counts == s_digit_counts:
            found_squares.add(num) # Add the unique number to the set

    # Print the total count of unique square numbers found.
    sys.stdout.write(str(len(found_squares)) + '
')

# Call the solve function when the script is executed.
if __name__ == '__main__':
    solve()