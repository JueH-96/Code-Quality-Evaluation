# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

# Count the number of '@' in both strings
count_S = S.count('@')
count_T = T.count('@')

# If the number of '@' is not the same, it's impossible to win
if count_S != count_T:
    print("No")
else:
    # Create a list of characters for both strings without '@'
    S_chars = [c for c in S if c != '@']
    T_chars = [c for c in T if c != '@']

    # Sort the lists of characters
    S_chars.sort()
    T_chars.sort()

    # Check if the sorted lists are the same
    if S_chars == T_chars:
        print("Yes")
    else:
        print("No")