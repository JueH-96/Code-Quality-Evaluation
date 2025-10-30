import sys

def solve():
    # Read N and M from the first line
    N, M = map(int, sys.stdin.readline().split())

    # Read the string S
    S = sys.stdin.readline().strip()

    # Convert S to a list of characters for mutability
    s_list = list(S)

    # Read the colors C
    C = list(map(int, sys.stdin.readline().split()))

    # Create a list of lists to group indices by color
    # color_groups[color_id] will contain a list of indices where characters of that color are found
    # Using 1-based indexing for colors (1 to M), so size M+1
    color_groups = [[] for _ in range(M + 1)]
    for i in range(N):
        color_groups[C[i]].append(i)

    # Perform operations for each color from 1 to M in order
    for color_id in range(1, M + 1):
        indices = color_groups[color_id]
        
        # If there are 0 or 1 characters of this color, a shift does not change anything.
        # The problem statement guarantees at least one char for each color, but checking len <= 1 is safe.
        if len(indices) <= 1:
            continue
        
        # 1. Collect the characters that are to be shifted
        # This captures their values *before* any modifications
        chars_to_shift = [s_list[idx] for idx in indices]
        
        # 2. Perform the right circular shift conceptually
        # The last character moves to the first position, others shift right by one
        shifted_chars = [chars_to_shift[-1]] + chars_to_shift[:-1]
        
        # 3. Place the shifted characters back into their respective positions in s_list
        for i in range(len(indices)):
            s_list[indices[i]] = shifted_chars[i]

    # Join the list of characters back into a string and print the result
    print("".join(s_list))

# Call the solve function to run the program
solve()