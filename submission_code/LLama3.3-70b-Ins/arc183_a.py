import itertools
import math

def generate_good_sequences(N, K):
    """
    Generate all good integer sequences of length NK where each integer from 1 to N appears exactly K times.
    
    Args:
    N (int): The number of distinct integers in the sequence.
    K (int): The number of times each integer appears in the sequence.
    
    Returns:
    list: A list of all good integer sequences.
    """
    # Generate a list of integers from 1 to N
    numbers = list(range(1, N + 1)) * K
    
    # Generate all permutations of the list
    permutations = list(itertools.permutations(numbers))
    
    return permutations

def lexicographical_order(permutations):
    """
    Sort the permutations in lexicographical order.
    
    Args:
    permutations (list): A list of permutations.
    
    Returns:
    list: The sorted list of permutations.
    """
    return sorted(permutations)

def find_sequence(N, K):
    """
    Find the floor((S+1)/2)-th good integer sequence in lexicographical order.
    
    Args:
    N (int): The number of distinct integers in the sequence.
    K (int): The number of times each integer appears in the sequence.
    
    Returns:
    list: The desired good integer sequence.
    """
    # Generate all good integer sequences
    permutations = generate_good_sequences(N, K)
    
    # Sort the permutations in lexicographical order
    permutations = lexicographical_order(permutations)
    
    # Calculate the index of the desired sequence
    S = len(permutations)
    index = math.floor((S + 1) / 2) - 1
    
    # Return the desired sequence
    return permutations[index]

def main():
    # Read the input from stdin
    N, K = map(int, input().split())
    
    # Find the desired sequence
    sequence = find_sequence(N, K)
    
    # Print the sequence
    print(' '.join(map(str, sequence)))

if __name__ == "__main__":
    main()