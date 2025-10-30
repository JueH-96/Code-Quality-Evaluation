import sys

def solve(S, X, Y):
    """
    Determine if there exists a string T (which can be empty) such that f(S,T,X)=f(S,T,Y).
    
    Parameters:
    S (str): a string consisting of lowercase English letters
    X (str): a string consisting of 0 and 1
    Y (str): a string consisting of 0 and 1
    
    Returns:
    bool: True if there exists a string T, False otherwise
    """
    # Initialize variables to store the number of 0s and 1s in X and Y
    x_zeros = X.count('0')
    x_ones = X.count('1')
    y_zeros = Y.count('0')
    y_ones = Y.count('1')
    
    # If the number of 0s in X is not equal to the number of 0s in Y, 
    # there cannot exist a string T that satisfies the condition
    if x_zeros != y_zeros:
        return False
    
    # If the number of 1s in X is not equal to the number of 1s in Y, 
    # there cannot exist a string T that satisfies the condition
    if x_ones != y_ones:
        return False
    
    # If the number of 0s in X is equal to the number of 0s in Y and 
    # the number of 1s in X is equal to the number of 1s in Y, 
    # there exists a string T that satisfies the condition
    return True

# Read the number of test cases
t = int(input())

# Iterate over each test case
for _ in range(t):
    # Read the strings S, X, and Y
    S = input()
    X = input()
    Y = input()
    
    # Determine if there exists a string T that satisfies the condition
    result = solve(S, X, Y)
    
    # Print the result
    if result:
        print("Yes")
    else:
        print("No")