MOD = 998244353

def count_sequences(N, M, K, X):
    # Calculate the total number of sequences of length N consisting of 1,...,K
    total_sequences = pow(K, N, MOD)

    # Calculate the number of sequences that can be obtained as a subsequence of any sequence of length N
    # This is equivalent to the number of ways to choose M positions out of N for the subsequence
    # and then choosing the values for those positions
    subsequence_count = pow(K, M, MOD) * pow(N, M, MOD) % MOD

    # The number of sequences that cannot be obtained as a subsequence is the total number of sequences
    # minus the number of sequences that can be obtained as a subsequence
    cannot_be_obtained = (total_sequences - subsequence_count) % MOD

    # The number of sequences that satisfy the condition is the number of sequences that cannot be obtained
    # as a subsequence of any sequence of length N
    return cannot_be_obtained

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
X = list(map(int, data[3:]))

# Calculate and print the result
result = count_sequences(N, M, K, X)
print(result)