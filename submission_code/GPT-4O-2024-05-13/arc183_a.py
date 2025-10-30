# YOUR CODE HERE
import itertools

def find_good_sequence(N, K):
    # Generate the initial sequence
    initial_sequence = [i for i in range(1, N + 1) for _ in range(K)]
    
    # Generate all permutations of the sequence
    all_permutations = sorted(set(itertools.permutations(initial_sequence)))
    
    # Calculate the index of the desired sequence
    index = (len(all_permutations) + 1) // 2 - 1
    
    # Get the desired sequence
    desired_sequence = all_permutations[index]
    
    # Print the desired sequence
    print(" ".join(map(str, desired_sequence)))

# Read input
import sys
input = sys.stdin.read
N, K = map(int, input().strip().split())

# Find and print the good sequence
find_good_sequence(N, K)