# YOUR CODE HERE
import sys

def solve():
    # Read the number of strings N
    N = int(sys.stdin.readline())
    
    # Read N strings and store them in a list S
    # .strip() is used to remove leading/trailing whitespace including the newline character
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Iterate through all possible pairs of indices (i, j)
    # The indices i and j correspond to the 1-based indices in the problem statement,
    # but are 0-based here due to Python list indexing (0 to N-1).
    for i in range(N):
        for j in range(N):
            # The problem requires distinct integers i and j.
            # In 0-based indexing, this means the indices must be different.
            if i == j:
                continue
            
            # Concatenate the strings S[i] and S[j] in this order
            concatenated_string = S[i] + S[j]
            
            # Check if the concatenated string is a palindrome.
            # A string T is a palindrome if it reads the same forwards and backwards.
            # In Python, this can be efficiently checked by comparing the string
            # with its reverse, obtained using slicing T[::-1].
            if concatenated_string == concatenated_string[::-1]:
                # If we find such a pair (i, j) that forms a palindrome,
                # print "Yes" and exit the function (and program).
                print("Yes")
                return

    # If the loops complete without finding any pair (i, j) such that
    # S[i] + S[j] is a palindrome, it means no such pair exists.
    # In this case, print "No".
    print("No")

# Call the solve function to execute the main logic of the program
solve()