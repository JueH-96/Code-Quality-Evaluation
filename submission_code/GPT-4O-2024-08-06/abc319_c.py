import itertools

def is_non_disappointing(permutation, lines):
    # Check each line for disappointment
    for line in lines:
        # Extract the indices of the line
        a, b, c = line
        # Check if the first two are the same and the third is different
        if permutation[a] == permutation[b] and permutation[a] != permutation[c]:
            return False
    return True

def main():
    import sys
    input = sys.stdin.read
    data = list(map(int, input().split()))
    
    # Define the lines to check: horizontal, vertical, and diagonal
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal lines
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical lines
        (0, 4, 8), (2, 4, 6)              # Diagonal lines
    ]
    
    # Total permutations
    total_permutations = 0
    non_disappointing_permutations = 0
    
    # Generate all permutations of the grid
    for permutation in itertools.permutations(data):
        total_permutations += 1
        if is_non_disappointing(permutation, lines):
            non_disappointing_permutations += 1
    
    # Calculate the probability
    probability = non_disappointing_permutations / total_permutations
    print(probability)