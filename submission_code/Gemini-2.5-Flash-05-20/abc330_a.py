# YOUR CODE HERE
import sys

def solve():
    # Read N and L from the first line
    N, L = map(int, sys.stdin.readline().split())

    # Read the scores A_i from the second line
    scores = list(map(int, sys.stdin.readline().split()))

    # Initialize a counter for people who passed
    passed_count = 0

    # Iterate through each score to check if it meets the passing criteria
    for score in scores:
        if score >= L:
            passed_count += 1

    # Print the total count of people who passed
    print(passed_count)

if __name__ == '__main__':
    solve()