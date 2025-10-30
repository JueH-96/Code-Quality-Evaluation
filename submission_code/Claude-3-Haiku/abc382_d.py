import sys

def soln():
    N, M = map(int, input().split())
    
    # Initialize an empty list to store the sequences
    sequences = []
    
    # Generate all possible sequences
    for a1 in range(1, M+1):
        for a2 in range(a1 + 10, M+1):
            for a3 in range(a2 + 10, M+1):
                if a3 <= M:
                    sequences.append((a1, a2, a3))
    
    # Sort the sequences in lexicographical order
    sequences.sort()
    
    # Print the number of sequences
    print(len(sequences))
    
    # Print each sequence
    for seq in sequences:
        print(*seq)

soln()