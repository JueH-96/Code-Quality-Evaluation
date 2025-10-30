# YOUR CODE HERE
from itertools import permutations

def is_valid_sequence(seq, grid):
    for i in range(0, 9, 3):  # Check rows
        if seq[i] == seq[i+1] and seq[i] != seq[i+2]:
            return False
    for i in range(3):  # Check columns
        if seq[i] == seq[i+3] and seq[i] != seq[i+6]:
            return False
    # Check diagonals
    if seq[0] == seq[4] and seq[0] != seq[8]:
        return False
    if seq[2] == seq[4] and seq[2] != seq[6]:
        return False
    return True

def main():
    import sys
    input = sys.stdin.read().split()
    grid = [list(map(int, input[i:i+3])) for i in range(0, 9, 3)]
    flat_grid = [item for sublist in grid for item in sublist]
    total_permutations = 0
    valid_permutations = 0
    
    for perm in permutations(flat_grid):
        total_permutations += 1
        if is_valid_sequence(perm, grid):
            valid_permutations += 1
    
    probability = valid_permutations / total_permutations
    print(f"{probability:.20f}")

if __name__ == "__main__":
    main()