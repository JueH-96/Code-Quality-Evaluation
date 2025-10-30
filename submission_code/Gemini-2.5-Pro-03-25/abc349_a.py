# YOUR CODE HERE
import sys

def solve():
    # Read the number of people, N
    n = int(sys.stdin.readline())

    # Read the final scores of the first N-1 people
    # The input line contains N-1 space-separated integers.
    # We use map(int, ...) to convert each string part to an integer,
    # and list(...) to convert the map object into a list.
    a = list(map(int, sys.stdin.readline().split()))

    # Calculate the sum of the final scores of the first N-1 people.
    # The sum() function efficiently calculates the sum of elements in the list.
    sum_a = sum(a)

    # The core idea is that the total score across all N people remains 0
    # throughout the games. Initially, everyone starts with 0 points, so the
    # total score is 0. Each game involves one person gaining 1 point and
    # another losing 1 point, so the total score change is +1 + (-1) = 0.
    # Therefore, the final total score must also be 0.
    # Let the final score of person N be score_n.
    # Then, A_1 + A_2 + ... + A_{N-1} + score_n = 0
    # which means sum_a + score_n = 0
    # Solving for score_n, we get:
    # score_n = -sum_a

    score_n = -sum_a

    # Print the calculated final score of person N.
    print(score_n)

# Execute the solve function to run the program logic.
solve()