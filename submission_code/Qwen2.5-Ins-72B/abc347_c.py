# YOUR CODE HERE
import sys

def is_possible(N, A, B, D):
    # Calculate the length of the week
    week_length = A + B
    
    # Check if all plans can be scheduled on holidays
    for i in range(N):
        # Calculate the day of the week for the i-th plan
        day_of_week = (D[i] - 1) % week_length + 1
        # If the day of the week is not a holiday, return "No"
        if day_of_week > A:
            return "No"
    
    # If all plans can be scheduled on holidays, return "Yes"
    return "Yes"

# Read input
input = sys.stdin.read
N, A, B, *D = map(int, input().split())

# Solve the problem
print(is_possible(N, A, B, D))