# YOUR CODE HERE
import sys

# Function to find all 1-blocks in a string
def find_blocks(S, N):
    """
    Finds all consecutive blocks of '1's in the binary string S.
    A block is defined by consecutive '1's surrounded by '0's or the string boundaries.
    Returns a list of tuples, where each tuple is (start_index, end_index) 
    of a block (0-based, inclusive).
    """
    blocks = []
    start = -1 # Tracks the start index of the current block, -1 if not currently in a block
    
    # Iterate through the string character by character
    for i in range(N):
        if S[i] == '1':
            # If we encounter a '1' and are not currently in a block, this is the start of a new block
            if start == -1: 
                start = i
        # If the current character is '0' and we were previously inside a block (start != -1)
        elif start != -1: 
            # The block ended at the previous index (i-1)
            blocks.append((start, i - 1))
            # Reset start index, indicating we are no longer in a block
            start = -1
            
    # After the loop, check if the string ended while we were inside a block
    # This happens if the string S ends with one or more '1's.
    if start != -1:
        # The last block extends to the end of the string (index N-1)
        blocks.append((start, N - 1))
        
    return blocks

def solve():
    # Read input N (length of S) and K (target block index, 1-based)
    # map(int, ...) converts the space-separated input values to integers
    # sys.stdin.readline() reads a line from standard input
    # .split() splits the line into parts based on whitespace
    N, K = map(int, sys.stdin.readline().split())
    
    # Read the binary string S from standard input
    # .strip() removes leading/trailing whitespace, including the newline character
    S = sys.stdin.readline().strip()

    # Find all 1-blocks in S using the helper function
    blocks = find_blocks(S, N)

    # The problem statement guarantees that S contains at least K 1-blocks.
    # Therefore, we can safely access the K-th and (K-1)-th blocks.
    
    # Get the (K-1)-th block and the K-th block from the list of blocks.
    # Remember K is 1-based, so we need indices K-2 and K-1 for the 0-based list 'blocks'.
    k_minus_1_block = blocks[K - 2]  # The (K-1)-th block in the sequence
    k_block = blocks[K - 1]          # The K-th block in the sequence

    # Extract the 0-based start and end indices for these two blocks.
    l_km1, r_km1 = k_minus_1_block  # Start and end index of the (K-1)-th block
    l_k, r_k = k_block              # Start and end index of the K-th block

    # Construct the new string T by rearranging parts of the original string S.
    # The core idea is to move the K-th block (S[l_k...r_k]) so that it immediately follows 
    # the (K-1)-th block (which ends at index r_km1).
    # The structure of the resulting string T will be:
    # [Part of S up to the end of block K-1] + [Block K] + [Part of S originally between block K-1 and block K] + [Part of S originally after block K]
    
    # Part 1: The substring of S from the beginning up to and including the end of the (K-1)-th block.
    # Python slicing S[a:b] extracts characters from index a up to (but not including) index b.
    # So, S[0 : r_km1 + 1] includes characters from index 0 to r_km1.
    part1 = S[0 : r_km1 + 1]
    
    # Part 2: The K-th block itself.
    # This is the substring S[l_k ... r_k].
    part2 = S[l_k : r_k + 1]
    
    # Part 3: The substring of S that was originally located between the end of the (K-1)-th block 
    # and the start of the K-th block. By definition of blocks, this segment consists only of '0's.
    # This corresponds to indices from r_km1 + 1 up to l_k - 1.
    # The slice S[r_km1 + 1 : l_k] captures this segment. If l_k = r_km1 + 1, this slice is empty.
    part3 = S[r_km1 + 1 : l_k]
    
    # Part 4: The substring of S that was originally located after the end of the K-th block.
    # This corresponds to indices from r_k + 1 up to N - 1.
    # The slice S[r_k + 1 : N] captures this segment. If r_k = N - 1, this slice is empty.
    part4 = S[r_k + 1 : N]

    # Combine these four parts in the specified order to form the resulting string T.
    # Using a list to hold the parts and then ''.join() is generally an efficient way 
    # to concatenate multiple string parts in Python, especially for large strings.
    T_list = [part1, part2, part3, part4]
    T = "".join(T_list)
    
    # Print the final constructed string T to standard output.
    print(T)

# Execute the main solution function when the script is run
solve()