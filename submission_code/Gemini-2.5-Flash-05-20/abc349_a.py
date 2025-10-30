# YOUR CODE HERE
import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    
    # Read the N-1 scores A_1, A_2, ..., A_{N-1}
    # These scores are given on a single line, space-separated.
    A_values = list(map(int, sys.stdin.readline().split()))
    
    # Calculate the sum of the known N-1 scores
    sum_of_known_scores = sum(A_values)
    
    # According to the problem's property (conservation of total score):
    # The initial total score is 0 (everyone starts with 0).
    # In each game, one player gains 1 point and another loses 1 point,
    # meaning the total sum of all scores in the system does not change.
    # Therefore, the final total sum of all scores must also be 0.
    # A_1 + A_2 + ... + A_{N-1} + A_N = 0
    # So, A_N = - (A_1 + A_2 + ... + A_{N-1})
    
    A_N = -sum_of_known_scores
    
    # Print the final score of person N
    print(A_N)

# Call the solve function to execute the logic
solve()