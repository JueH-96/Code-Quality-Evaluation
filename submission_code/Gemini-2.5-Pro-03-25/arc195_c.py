# YOUR CODE HERE
import sys

# Increase recursion depth for safety, although not strictly needed for this iterative solution
# sys.setrecursionlimit(2000) 

def solve():
    """
    Solves a single test case. Reads R and B, determines if a valid cycle placement exists,
    and prints the result according to the specified format.
    """
    R, B = map(int, sys.stdin.readline().split())

    # Necessary condition: The number of Red pieces (R) must be even.
    # This is derived from parity arguments: Red moves flip the parity of r+c, Blue moves preserve it.
    # To return to the starting square's parity after traversing the cycle, the total number of parity flips (equal to R) must be even.
    if R % 2 != 0:
        print("No")
        return

    # If R is even, a solution is possible. We print "Yes" and provide one such construction.
    print("Yes")
    
    # Use a large coordinate base K to avoid negative coordinates during relative coordinate calculations
    # and to ensure final coordinates are well within the 1 to 10^9 range.
    # K = 10^5 is chosen somewhat arbitrarily but is large enough for typical constraints.
    # It provides a large baseline offset. Additional shifting might be applied if relative coordinates become too negative.
    K = 10**5 
    
    # The construction strategy is based on extending a known valid base structure for R=2, B=1.
    # Base structure uses relative coordinates: P1=R(0,0), P2=B(0,1), P3=R(1,0).
    # The cycle path is P1 -> P2 -> P3 -> P1.
    # P1(R) moves to P2(B) position (0,1). R move: (0,0) -> (0,1) (vertical). Correct.
    # P2(B) moves to P3(R) position (1,0). B move: (0,1) -> (1,0) (diagonal). Correct.
    # P3(R) moves to P1(R) position (0,0). R move: (1,0) -> (0,0) (horizontal). Correct.
    base_pieces_relative = [('R', 0, 0), ('B', 0, 1), ('R', 1, 0)]
    
    # The list `current_pieces` will store the sequence of pieces (type, r, c) using relative coordinates.
    # It starts with the base structure.
    current_pieces = list(base_pieces_relative)
    
    # If B > 1, insert B-1 additional Blue pieces into the sequence.
    # The insertion happens between P2 and P3. The original move P2 -> P3 is a valid B move.
    # The sequence becomes P2 -> B_1 -> B_2 -> ... -> B_{B-1} -> P3. Each step must be a valid B move.
    
    # The insertion index points to where the next Blue piece should be inserted (before the original P3).
    insert_idx_B = 2 
    # `curr_pos_B` keeps track of the position of the last piece added (initially P2's position).
    curr_pos_B = base_pieces_relative[1][1:] # Position of P2 = (0,1)
    
    for i in range(B - 1):
        # Calculate the position for the next Blue piece using a diagonal move.
        # The move (-1, +1) relative to the current position is chosen. This expands the path outwards.
        next_pos_B = (curr_pos_B[0] - 1, curr_pos_B[1] + 1) 
        current_pieces.insert(insert_idx_B, ('B', next_pos_B[0], next_pos_B[1]))
        insert_idx_B += 1 # Move insertion index forward
        curr_pos_B = next_pos_B # Update current position

    # If R > 2, insert R-2 additional Red pieces (k = (R-2)//2 pairs of RR).
    # The insertion happens after the original P3 piece, just before the cycle closes back to P1.
    # The sequence becomes ... P3 -> R_1 -> R_2 -> ... -> R_{R-2} -> P1. Each step must be a valid R move.
    
    # Find the index of the original P3 piece ('R', 1, 0) in the list. Its index might have changed if B > 1.
    original_P3_idx = -1
    for idx, p in enumerate(current_pieces):
        if p == ('R', 1, 0):
            original_P3_idx = idx
            break
    
    # If R > 2, proceed with insertions.
    insert_idx_R = original_P3_idx + 1 # Insert after the original P3 piece.
    curr_pos_R = current_pieces[original_P3_idx][1:] # Position of original P3 = (1,0)

    k = (R - 2) // 2 # Number of RR pairs to insert
    for i in range(k):
        # Each iteration inserts two R pieces forming a small L-shape step pattern.
        # This pattern ensures valid R moves between consecutive pieces and maintains distinct coordinates.
        
        # First R piece: moves vertically down from `curr_pos_R`.
        pos1 = (curr_pos_R[0], curr_pos_R[1] - 1) 
        current_pieces.insert(insert_idx_R, ('R', pos1[0], pos1[1]))
        insert_idx_R += 1
        
        # Second R piece: moves horizontally left from `pos1`.
        pos2 = (pos1[0] - 1, pos1[1]) 
        current_pieces.insert(insert_idx_R, ('R', pos2[0], pos2[1]))
        insert_idx_R += 1
        curr_pos_R = pos2 # Update current position to the last added piece's position


    # Calculate the minimum row and column values among all generated relative coordinates.
    min_r, min_c = float('inf'), float('inf')
    if current_pieces: # Check if the list is not empty
       min_r = current_pieces[0][1]
       min_c = current_pieces[0][2]

    for _, r, c in current_pieces:
        min_r = min(min_r, r)
        min_c = min(min_c, c)
    
    # Calculate the necessary coordinate shifts to ensure all final coordinates are positive (>= 1).
    # Start with a base shift of K. If relative coordinates resulted in non-positive values,
    # increase the shift to guarantee positivity.
    shift_r, shift_c = K, K
    if min_r <= 0: 
        # Calculate additional shift needed for rows
        shift_r = K + (1 - min_r)
    if min_c <= 0: 
        # Calculate additional shift needed for columns
        shift_c = K + (1 - min_c)

    # Print the final list of pieces with their types and absolute coordinates.
    for p_type, r, c in current_pieces:
        # Apply the calculated shifts to the relative coordinates
        final_r = r + shift_r
        final_c = c + shift_c
        # Final coordinates are guaranteed to be >= 1 and within board limits due to large K.
        print(f"{p_type} {final_r} {final_c}")


# Read the number of test cases
T = int(sys.stdin.readline())
# Process each test case
for _ in range(T):
    solve()