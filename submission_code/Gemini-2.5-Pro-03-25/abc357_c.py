# YOUR CODE HERE
import sys

# Use a dictionary for memoization to store computed carpets
# This helps avoid recomputing the same carpet level multiple times if needed,
# and makes the recursive structure efficient.
memo = {}

def carpet(k):
    """
    Generates a level-k Sierpinski carpet recursively using memoization.
    
    A level-k carpet is defined as follows:
    - Level-0: A 1x1 grid with a single black cell ('#').
    - Level-k (k > 0): A 3^k x 3^k grid. When divided into nine 3^(k-1) x 3^(k-1) blocks:
        - The central block is all white ('.').
        - The other eight blocks are level-(k-1) carpets.

    Args:
        k: The level of the carpet (non-negative integer).
        
    Returns:
        A list of strings representing the level-k carpet. Each string is a row.
    """
    # Base case: Level 0 carpet is a single black cell '#'
    if k == 0:
        return ['#']
        
    # Check the memoization cache first to see if level k carpet is already computed
    if k in memo:
        return memo[k]
        
    # If not in cache, compute it recursively:
    # 1. Get the carpet for the previous level (k-1)
    prev_carpet = carpet(k-1)
    
    # 2. Determine the side length of the previous level carpet (which is 3^(k-1))
    size_prev = len(prev_carpet) 
    
    # 3. Initialize the list to store the rows of the new level-k carpet
    new_carpet = []
    
    # 4. Construct the 3x3 grid of blocks based on the definition:
    
    # Top row of blocks: Consists of three copies of the level-(k-1) carpet horizontally.
    # Iterate through each row of the previous carpet and triplicate it.
    for i in range(size_prev):
        row_str = prev_carpet[i] + prev_carpet[i] + prev_carpet[i]
        new_carpet.append(row_str)
        
    # Middle row of blocks: Consists of a level-(k-1) carpet, a white block, and another level-(k-1) carpet.
    # Create a string representing a row of the central white block (all '.')
    white_block_row_part = '.' * size_prev
    # Iterate through each row of the previous carpet and construct the middle rows.
    for i in range(size_prev):
        row_str = prev_carpet[i] + white_block_row_part + prev_carpet[i]
        new_carpet.append(row_str)
        
    # Bottom row of blocks: Similar to the top row, consists of three copies of the level-(k-1) carpet.
    # Iterate through each row of the previous carpet and triplicate it.
    for i in range(size_prev):
        row_str = prev_carpet[i] + prev_carpet[i] + prev_carpet[i]
        new_carpet.append(row_str)
        
    # 5. Store the computed carpet in the memoization cache before returning
    memo[k] = new_carpet
    
    # 6. Return the generated level-k carpet
    return new_carpet

# --- Main execution ---

if __name__ == "__main__":
    # Read the integer N (the desired carpet level) from standard input
    N = int(sys.stdin.readline())

    # Generate the level N carpet using the recursive function
    result_carpet = carpet(N)

    # Print the resulting carpet to standard output, row by row
    for row in result_carpet:
        print(row)