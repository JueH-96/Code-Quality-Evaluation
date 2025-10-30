import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Condition 1: The length of the string T (|T|) must be odd.
    if N % 2 == 0:
        print("No")
        return

    # Calculate the 0-indexed position of the middle character.
    # For an odd length N (e.g., N=5, indices 0,1,2,3,4), the middle index is N // 2 (5 // 2 = 2).
    # For N=1, the middle index is 0 (1 // 2 = 0).
    mid_idx = N // 2

    # Condition 3: The character at the middle position must be '/'.
    # In 1-indexed terms, this is the ((|T|+1)/2)-th character.
    # In 0-indexed terms, this is S[mid_idx].
    if S[mid_idx] != '/':
        print("No")
        return

    # Condition 2: The characters before the middle ('/') must all be '1'.
    # These are characters from index 0 up to (but not including) mid_idx.
    # In 1-indexed terms, this is the 1-st through ((|T|+1)/2 - 1)-th characters.
    for i in range(mid_idx): # Iterates from 0 to mid_idx - 1
        if S[i] != '1':
            print("No")
            return

    # Condition 4: The characters after the middle ('/') must all be '2'.
    # These are characters from index mid_idx + 1 up to (and including) N - 1.
    # In 1-indexed terms, this is the ((|T|+1)/2 + 1)-th through |T|-th characters.
    for i in range(mid_idx + 1, N): # Iterates from mid_idx + 1 to N - 1
        if S[i] != '2':
            print("No")
            return

    # If all conditions are satisfied, the string is an 11/22 string.
    print("Yes")

# Call the solve function to run the program
solve()