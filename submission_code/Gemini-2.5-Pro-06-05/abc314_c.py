import sys

def solve():
    """
    Reads input, performs the described character shifts by color, and prints the result.
    """
    # Read problem parameters from standard input.
    # N is the length of the string, M is the number of colors.
    try:
        N, M = map(int, sys.stdin.readline().split())
    except ValueError:
        # Handles potential empty line at the end of input, common in some environments.
        return

    S = sys.stdin.readline().strip()
    C = list(map(int, sys.stdin.readline().split()))

    # --- Stage 1: Preprocessing ---
    # Group indices by their assigned color.
    # `color_indices[i]` will store a list of 0-based indices for characters of color i+1.
    # This O(N) pass allows for efficient access to all characters of a specific color later.
    color_indices = [[] for _ in range(M)]
    for j in range(N):
        # Input colors are 1-based, so we subtract 1 for 0-based list indexing.
        color_idx = C[j] - 1
        color_indices[color_idx].append(j)

    # Convert the string to a list of characters to allow for mutable item assignment.
    s_list = list(S)

    # --- Stage 2: Stateful Shifting ---
    # The operations are performed sequentially for each color from 1 to M.
    for i in range(M):
        indices = color_indices[i]
        
        # A circular shift requires at least two elements.
        if len(indices) <= 1:
            continue
        
        # Perform a right circular shift on the characters for the current color group.
        # This is done in-place for memory efficiency.
        
        # First, save the character from the last position in this color group.
        last_char = s_list[indices[-1]]
        
        # Next, shift all other characters one position to the right within the group.
        # We iterate backwards from the second-to-last to the first element to avoid
        # overwriting characters before they have been moved.
        for k in range(len(indices) - 1, 0, -1):
            s_list[indices[k]] = s_list[indices[k-1]]
        
        # Finally, place the saved `last_char` into the first position of the group.
        s_list[indices[0]] = last_char

    # --- Final Output ---
    # Join the characters in the list back into a single string and print to standard output.
    print("".join(s_list))

solve()