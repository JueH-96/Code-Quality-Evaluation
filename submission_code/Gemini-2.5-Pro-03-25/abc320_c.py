# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    """
    Solves the slot machine problem. Reads input, computes the minimum time
    to stop all reels displaying the same character, and prints the result.
    """
    
    # Read the integer M, the length of the reel strings
    M = int(sys.stdin.readline())
    
    # Read the three reel strings S_1, S_2, S_3
    S = [sys.stdin.readline().strip() for _ in range(3)]

    # Precompute the positions (indices) of each character '0' through '9' for each reel.
    # pos[i][c] will store a list of indices k such that S[i][k] == c.
    # Using defaultdict simplifies handling cases where a character might not appear in a reel.
    pos = [defaultdict(list) for _ in range(3)]
    for i in range(3):  # For each reel i (0, 1, 2)
        for k in range(M):  # For each position k (0 to M-1) in the string S[i]
            # Store the index k for the character S[i][k] found in reel i
            pos[i][S[i][k]].append(k)

    # Initialize the minimum total time found across all characters to infinity.
    # float('inf') is a convenient representation for infinity in Python.
    min_total_time = float('inf')

    # Iterate through all possible target characters, from '0' to '9'.
    for char_code in range(ord('0'), ord('9') + 1):
        c = chr(char_code)  # Get the character corresponding to the current char_code
        
        # Check if the character c is present in all three reels.
        # If a character is not present in any reel, it's impossible to align all reels on it.
        # An empty list (returned by defaultdict for a missing key) evaluates to False in boolean context.
        if not pos[0][c] or not pos[1][c] or not pos[2][c]:
            continue  # Skip to the next character if c is not present in all reels

        # Initialize the minimum time required to align on this specific character c to infinity.
        min_time_c = float('inf')
        
        # Retrieve the lists of indices where character c appears for each reel.
        P1 = pos[0][c]
        P2 = pos[1][c]
        P3 = pos[2][c]

        # Iterate through all combinations of indices (k1, k2, k3), one index from each reel's list.
        # k_i represents an index in reel i where character c appears (S[i][k_i] == c).
        for k1 in P1:
            for k2 in P2:
                for k3 in P3:
                    
                    # Calculate the minimum possible maximum stop time for this combination of indices (k1, k2, k3).
                    # The calculation depends on whether the indices k1, k2, k3 are distinct or not.
                    # This ensures that the corresponding stop times t1, t2, t3 can be chosen distinct.
                    
                    current_time = 0 # Temporary variable to store calculated time for this (k1, k2, k3) combo
                    
                    if k1 == k2 == k3:
                        # Case 3: All three indices are identical (let k = k1 = k2 = k3).
                        # To get distinct stop times, we need times like k, k+M, k+2M.
                        # The maximum of these times is k + 2*M.
                        current_time = k1 + 2 * M
                    elif k1 == k2:
                        # Case 2: Exactly two indices are identical (k1=k2=k), the third is different (k3).
                        # We need distinct times t1, t2, t3. Smallest possible distinct times are k, k+M, k3.
                        # The maximum of these times is k + M (since k, k3 < M).
                        current_time = k1 + M
                    elif k1 == k3:
                        # Case 2: Similarly, if k1=k3=k and k2 is different.
                        # Smallest distinct times are k, k2, k+M. Max time is k + M.
                        current_time = k1 + M
                    elif k2 == k3:
                        # Case 2: Similarly, if k2=k3=k and k1 is different.
                        # Smallest distinct times are k1, k, k+M. Max time is k + M.
                        current_time = k2 + M
                    else:
                        # Case 1: All three indices k1, k2, k3 are distinct.
                        # We can stop the reels at times t1=k1, t2=k2, t3=k3. These are already distinct.
                        # The maximum time is simply max(k1, k2, k3).
                        current_time = max(k1, k2, k3)

                    # Update the minimum time found so far for aligning on character c.
                    min_time_c = min(min_time_c, current_time)

        # After checking all index combinations for character c, update the overall minimum time found.
        min_total_time = min(min_total_time, min_time_c)

    # After checking all characters '0' through '9':
    # If min_total_time is still infinity, it means no character allowed alignment. Impossible.
    if min_total_time == float('inf'):
        print("-1")
    else:
        # Otherwise, print the overall minimum time found.
        print(min_total_time)

# Call the main function to solve the problem
solve()