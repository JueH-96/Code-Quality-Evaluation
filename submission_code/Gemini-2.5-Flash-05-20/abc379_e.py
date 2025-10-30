import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    total_sum = 0

    # current_ones_sum will hold the value (10^L - 1) // 9,
    # where L is the length of the suffix S[k...N-1].
    # For k=0, L=N. So, it starts with (10^N - 1) // 9.
    # As k increments, L (N-k) decrements by 1.
    # We use the relationship: ones_sum(L-1) = (ones_sum(L) - 1) // 10
    
    # Calculate initial current_ones_sum for k=0 (length L=N)
    # This represents 1 + 10 + ... + 10^(N-1)
    current_ones_sum = (pow(10, N) - 1) // 9

    # Iterate through each digit S[k] (0-indexed)
    for k in range(N):
        digit_val = int(S[k])

        # Contribution of S[k] to the total sum:
        # digit_val * (number of possible starting positions 'i' (0-indexed) such that i <= k)
        #           * (sum of all possible place values for S[k] within its substrings)
        # Number of possible starting positions 'i' is (k+1).
        # Sum of possible place values for S[k] is current_ones_sum (which is (10^(N-k) - 1) // 9).
        total_sum += digit_val * (k + 1) * current_ones_sum

        # Update current_ones_sum for the next iteration (k+1).
        # For k+1, the suffix length will be N-(k+1), which is (N-k)-1.
        # This update should only happen if there is a next iteration (i.e., k is not N-1).
        if k < N - 1:
            current_ones_sum = (current_ones_sum - 1) // 10

    sys.stdout.write(str(total_sum) + '
')

solve()