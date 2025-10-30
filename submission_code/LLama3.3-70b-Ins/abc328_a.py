import sys

def calculate_total_score(N, X, scores):
    """
    Calculate the total score for all problems with a score of X or less.

    Args:
    N (int): The number of problems.
    X (int): The maximum score.
    scores (list): A list of scores for each problem.

    Returns:
    int: The total score for all problems with a score of X or less.
    """
    total_score = 0
    for score in scores:
        if score <= X:
            total_score += score
    return total_score

def main():
    # Read input from stdin
    N, X = map(int, sys.stdin.readline().split())
    scores = list(map(int, sys.stdin.readline().split()))

    # Calculate and print the total score
    total_score = calculate_total_score(N, X, scores)
    print(total_score)

if __name__ == "__main__":
    main()