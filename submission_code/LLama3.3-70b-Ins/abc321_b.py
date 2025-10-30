import sys

def calculate_min_score(N, X, A):
    """
    Calculate the minimum score needed in the last round to achieve a final grade of X or higher.

    Args:
    N (int): The total number of rounds.
    X (int): The desired final grade.
    A (list): A list of scores earned in the first N-1 rounds.

    Returns:
    int: The minimum score needed in the last round. Returns -1 if it's impossible to achieve the desired final grade.
    """
    # Sort the scores in ascending order
    A.sort()

    # Calculate the sum of the scores excluding the highest and lowest
    current_sum = sum(A[1:-1])

    # Calculate the minimum score needed in the last round
    for score in range(101):
        # Create a copy of the list of scores
        scores = A.copy()
        scores.append(score)
        scores.sort()

        # Calculate the sum of the scores excluding the highest and lowest
        new_sum = sum(scores[1:-1])

        # Check if the new sum is greater than or equal to the desired final grade
        if new_sum >= X:
            return score

    # If no score can achieve the desired final grade, return -1
    return -1

def main():
    # Read the input from stdin
    N, X = map(int, input().split())
    A = list(map(int, input().split()))

    # Calculate the minimum score needed in the last round
    min_score = calculate_min_score(N, X, A)

    # Print the result
    print(min_score)

if __name__ == "__main__":
    main()