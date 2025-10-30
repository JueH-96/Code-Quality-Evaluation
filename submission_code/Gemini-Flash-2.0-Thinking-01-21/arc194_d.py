import sys
from collections import Counter

# Define the modulus (not needed for this simplified logic, but included based on initial analysis)
# MOD = 998244353

# Function to find prime factorization of a VPS
def prime_factorization(s):
    """
    Finds the prime factorization of a valid parenthesis sequence s.
    A prime factor is a non-empty valid parenthesis sequence that cannot
    be split into two non-empty valid parenthesis sequences.
    """
    factors = []
    n = len(s)
    i = 0
    balance = 0
    for j in range(n):
        if s[j] == '(':
            balance += 1
        else:
            balance -= 1
        if balance == 0:
            # Found a non-empty prefix s[i...j] that is a valid parenthesis sequence.
            # By construction of the iteration, this must be the next prime factor.
            factors.append(s[i : j+1])
            i = j + 1
    return factors

# Read input
N = int(sys.stdin.readline())
S = sys.stdin.readline().strip()

# Get prime factors of the input string S
factors = prime_factorization(S)

# According to the refined hypothesis based on sample behavior,
# the only effective operation is applying the reversal on the entire string S.
# This operation transforms the sequence of prime factors S = S_1 S_2 ... S_k
# into S_k S_{k-1} ... S_1.
# Repeated application of this operation only alternates between the original
# sequence and the fully reversed sequence.
# The number of distinct strings reachable is 1 if the original sequence
# of factors is the same as the reversed sequence (i.e., the sequence is a palindrome),
# and 2 otherwise.

# Check if the sequence of factors is a palindrome
is_palindrome = True
k = len(factors)
for i in range(k // 2):
    # Compare factor string from the beginning with corresponding factor string from the end
    if factors[i] != factors[k - 1 - i]:
        is_palindrome = False
        break

# The number of distinct strings is 1 if the factor sequence is a palindrome, and 2 otherwise.
if is_palindrome:
    print(1)
else:
    print(2)