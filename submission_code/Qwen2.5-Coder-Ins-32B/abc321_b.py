# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = int(input[1])
    A = list(map(int, input[2:]))

    # We need to find the minimum score in the N-th round to achieve a final grade of X or higher
    # The final grade is the sum of N-2 scores excluding the highest and lowest

    # Sort the current scores
    A.sort()

    # Calculate the sum of the N-2 middle scores
    current_sum = sum(A[1:N-2])

    # We need to find the minimum score y such that (current_sum + y - min(max(A), y) - max(min(A), y)) >= X
    # This simplifies to finding the minimum y such that (current_sum + y - max(A) - min(A)) >= X if y is not the new min or max
    # Or (current_sum + y - max(A) - y) >= X if y is the new min
    # Or (current_sum + y - y - min(A)) >= X if y is the new max
    # Which simplifies to finding the minimum y such that (current_sum - max(A) - min(A) + y) >= X

    # Calculate the required additional score
    required_score = X - (current_sum - A[-1] - A[0])

    # The required score must be between 0 and 100
    if required_score > 100:
        print(-1)
    elif required_score < 0:
        print(0)
    else:
        print(required_score)

if __name__ == "__main__":
    main()