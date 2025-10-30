import sys
import itertools

def solve(N, K, R):
    """
    Generate all integer sequences of length N that satisfy the conditions.

    Args:
    N (int): The length of the sequence.
    K (int): The sum of the sequence must be a multiple of K.
    R (list): A list of integers where the i-th element is between 1 and R_i, inclusive.

    Returns:
    list: A list of sequences that satisfy the conditions.
    """
    # Generate all possible sequences
    sequences = list(itertools.product(*[range(1, r + 1) for r in R]))

    # Filter sequences that sum to a multiple of K
    valid_sequences = [seq for seq in sequences if sum(seq) % K == 0]

    # Sort the sequences in lexicographical order
    valid_sequences.sort()

    return valid_sequences

def main():
    # Read input from stdin
    N, K = map(int, input().split())
    R = list(map(int, input().split()))

    # Solve the problem
    sequences = solve(N, K, R)

    # Print the answer to stdout
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()