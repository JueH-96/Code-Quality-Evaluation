import itertools
import sys

def calculate_probability(grid):
    # Flatten the grid to a list of numbers
    numbers = [grid[i][j] for i in range(3) for j in range(3)]
    
    # Total permutations of the numbers
    total_permutations = list(itertools.permutations(numbers))
    total_count = len(total_permutations)
    
    # Count valid permutations
    valid_count = 0
    
    # Check each permutation
    for perm in total_permutations:
        disappointed = False
        
        # Check rows
        for i in range(3):
            if perm[i*3] == perm[i*3 + 1] and perm[i*3 + 1] != perm[i*3 + 2]:
                disappointed = True
                break
        
        # Check columns
        if not disappointed:
            for j in range(3):
                if perm[j] == perm[j + 3] and perm[j + 3] != perm[j + 6]:
                    disappointed = True
                    break
        
        # Check diagonals
        if not disappointed:
            if perm[0] == perm[4] and perm[4] != perm[8]:
                disappointed = True
            elif perm[6] == perm[4] and perm[4] != perm[2]:
                disappointed = True
        
        if not disappointed:
            valid_count += 1
    
    # Calculate probability
    probability = valid_count / total_count
    return probability

def main():
    input_data = sys.stdin.read().strip().split()
    grid = [[int(input_data[i * 3 + j]) for j in range(3)] for i in range(3)]
    
    probability = calculate_probability(grid)
    print(f"{probability:.30f}")

if __name__ == "__main__":
    main()