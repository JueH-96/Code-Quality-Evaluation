import sys

def get_digit_counts(s):
    """Calculates frequency of each digit (0-9) in a string."""
    counts = [0] * 10
    for char in s:
        counts[int(char)] += 1
    return counts

# Read input
N = int(sys.stdin.readline())
S = sys.stdin.readline().strip()

# Compute digit counts for the input string S
counts_S = get_digit_counts(S)

# Set to store found square numbers
found_squares = set()

# Iterate through possible square roots k
# We need k*k < 10**N.
# The maximum value formed by permuting N digits is less than 10**N.
# So any square number k*k that is a candidate must be less than 10**N.
# The maximum k to check is floor(sqrt(10**N - 1)).
# We can loop while k*k is less than 10**N.
# The max value of k is roughly sqrt(10^13) approx 3.16e6 for N=13.
# The loop will run around 3.2 million times for N=13.

k = 0
while True:
    num = k * k
    
    # If the square exceeds the maximum possible value representable by N digits, stop.
    # The maximum value formed by permuting digits of S is less than 10^N.
    # So we stop when k*k >= 10**N.
    # For N=1, 10**1 = 10. Squares are 0, 1, 4, 9. The next square is 16 which >= 10.
    # This condition correctly stops the loop.
    if num >= 10**N:
         break
    
    # Convert the square number to string
    T_str = str(num)
    
    # Calculate digit counts for the square number string
    counts_T = get_digit_counts(T_str)
    
    # The number of digits in T_str must be at most N.
    # This is guaranteed by the loop condition num < 10**N.
    # If len(T_str) < N, the number num can only be formed by a permutation of S
    # if the remaining N - len(T_str) digits in S are all zeros.
    # We check this by padding counts_T with required leading zeros.
    required_leading_zeros = N - len(T_str)
    
    # Create a temporary counts list by adding required leading zeros to counts_T
    temp_counts = list(counts_T)
    
    # required_leading_zeros should be non-negative since len(T_str) <= N
    temp_counts[0] += required_leading_zeros
    
    # Compare the temporary counts with the counts of digits in S
    if temp_counts == counts_S:
        found_squares.add(num)
    
    k += 1

# Print the number of distinct square numbers found
print(len(found_squares))