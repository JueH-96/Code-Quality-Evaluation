import itertools
import sys
input = sys.stdin.read

def main():
    # Read input
    data = input().strip().split()
    grid = [list(map(int, data[i*3:(i+1)*3])) for i in range(3)]
    
    # All possible lines in the grid
    lines = [
        grid[0], grid[1], grid[2],  # horizontal lines
        [grid[0][0], grid[1][0], grid[2][0]],  # vertical lines
        [grid[0][1], grid[1][1], grid[2][1]],
        [grid[0][2], grid[1][2], grid[2][2]],
        [grid[0][0], grid[1][1], grid[2][2]],  # diagonal lines
        [grid[2][0], grid[1][1], grid[0][2]]
    ]
    
    # Total permutations of viewing all cells
    total_permutations = 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
    
    # Count valid permutations where no disappointment occurs
    valid_permutations = 0
    
    # Generate all permutations of the grid indices
    for perm in itertools.permutations(range(9)):
        # Map permutation to grid values
        perm_values = [grid[i//3][i%3] for i in perm]
        
        # Check if this permutation causes disappointment
        disappointed = False
        for line in lines:
            # Get the indices in the permutation for the current line
            indices = [perm.index(i*3 + j) for i, j in enumerate(line)]
            values = [perm_values[idx] for idx in indices]
            
            # Check if the first two are the same and the third is different
            if values[0] == values[1] and values[0] != values[2]:
                disappointed = True
                break
        
        if not disappointed:
            valid_permutations += 1
    
    # Calculate the probability of not getting disappointed
    probability = valid_permutations / total_permutations
    
    # Print the result with high precision
    print(f"{probability:.30f}")

if __name__ == "__main__":
    main()