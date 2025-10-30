# YOUR CODE HERE
def find_minimum_operations(S, T):
    # Initialize the result list
    result = []
    
    # Convert strings to lists for mutability
    S_list = list(S)
    T_list = list(T)
    
    # Iterate over each character in the strings
    for i in range(len(S_list)):
        # If the characters at the current position are different
        if S_list[i] != T_list[i]:
            # Change the character in S to match T
            S_list[i] = T_list[i]
            # Append the current state of S to the result list
            result.append(''.join(S_list))
    
    # Print the number of operations
    print(len(result))
    # Print each state in the result list
    for state in result:
        print(state)

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()

S = data[0]
T = data[1]

# Call the function with the input strings
find_minimum_operations(S, T)