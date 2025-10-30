import itertools
import sys

def solve():
    # Read N and K from the first line of input
    N, K = map(int, sys.stdin.readline().split())

    # Read the sequence A from the second line of input
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize max_xor_sum to 0.
    # Since all A_i are non-negative, any XOR sum will also be non-negative.
    max_xor_sum = 0

    # Use itertools.combinations to generate all combinations of K distinct elements.
    # The constraint C(N, K) <= 10^6 ensures that this loop is efficient enough.
    for combination in itertools.combinations(A, K):
        current_xor_sum = 0
        
        # Calculate the XOR sum for the current combination
        for element in combination:
            current_xor_sum ^= element
        
        # Update the overall maximum XOR sum if the current one is greater
        if current_xor_sum > max_xor_sum:
            max_xor_sum = current_xor_sum
            
    # Print the final maximum XOR sum to standard output
    sys.stdout.write(str(max_xor_sum) + "
")

# This check ensures that solve() is called only when the script is executed directly
if __name__ == '__main__':
    solve()