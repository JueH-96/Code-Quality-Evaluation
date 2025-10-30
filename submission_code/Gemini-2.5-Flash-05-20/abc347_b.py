import sys

def solve():
    S = sys.stdin.readline().strip()
    
    unique_substrings = set()
    
    n = len(S)
    
    # Iterate through all possible starting indices (i)
    for i in range(n):
        # Iterate through all possible ending indices (j) for the current starting index.
        # j goes from i+1 to n to ensure the substring is non-empty.
        # S[i:j] extracts the substring from index i up to (but not including) index j.
        for j in range(i + 1, n + 1):
            substring = S[i:j]
            unique_substrings.add(substring)
            
    # The size of the set gives the number of unique non-empty substrings.
    print(len(unique_substrings))

# Call the solve function to run the program
solve()