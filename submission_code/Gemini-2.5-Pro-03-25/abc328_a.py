import sys

def solve():
    # Read the first line containing N and X
    n, x = map(int, sys.stdin.readline().split())

    # Read the second line containing the scores S_1, S_2, ..., S_N
    scores = list(map(int, sys.stdin.readline().split()))

    # Initialize the total score
    total_score = 0

    # Iterate through each score
    for score in scores:
        # Check if the score is less than or equal to X
        if score <= x:
            # Add the score to the total
            total_score += score

    # Print the final total score
    print(total_score)

# Call the solve function to execute the logic
solve()