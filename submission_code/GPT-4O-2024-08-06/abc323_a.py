# YOUR CODE HERE
def check_even_positions(S):
    # Check every even index (1-based) which corresponds to odd index (0-based)
    for i in range(1, 16, 2):
        if S[i] != '0':
            print("No")
            return
    print("Yes")

# Read input
import sys
input = sys.stdin.read
S = input().strip()

# Call the function to check the condition and print the result
check_even_positions(S)