# YOUR CODE HERE
import sys

def solve(S, X, Y):
    # Count the number of 0s and 1s in X and Y
    count_X = [X.count('0'), X.count('1')]
    count_Y = [Y.count('0'), Y.count('1')]

    # If the counts are the same, T can be an empty string
    if count_X == count_Y:
        return "Yes"
    
    # If S is empty, T can be any string
    if not S:
        return "Yes"
    
    # If the counts are different, check if the difference can be made up by T
    diff = [count_X[0] - count_Y[0], count_X[1] - count_Y[1]]
    if diff[0] == 0 and diff[1] == 0:
        return "Yes"
    if diff[0] == 0 or diff[1] == 0:
        return "No"
    
    # Check if the difference can be made up by T
    if abs(diff[0]) == abs(diff[1]):
        return "Yes"
    return "No"

t = int(input())
for _ in range(t):
    S = input().strip()
    X = input().strip()
    Y = input().strip()
    print(solve(S, X, Y))