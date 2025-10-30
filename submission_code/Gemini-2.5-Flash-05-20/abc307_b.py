import sys

def is_palindrome(s: str) -> bool:
    """
    Checks if a given string is a palindrome.
    A string is a palindrome if it reads the same forwards and backwards.
    """
    return s == s[::-1]

def solve():
    """
    Reads N strings and determines if there are distinct integers i and j
    such that the concatenation of S_i and S_j is a palindrome.
    """
    N = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Iterate through all distinct pairs (i, j)
    for i in range(N):
        for j in range(N):
            # Ensure i and j are distinct as per problem statement
            if i == j:
                continue
            
            # Concatenate the strings S_i and S_j
            concatenated_string = S[i] + S[j]
            
            # Check if the concatenated string is a palindrome
            if is_palindrome(concatenated_string):
                print("Yes")
                return # Found a pair, so we can exit immediately

    # If no such pair is found after checking all possibilities
    print("No")

# Call the solve function to run the program
solve()