import sys

def calculate_score(A, S, i):
    """Calculate the total score of player i"""
    score = 0
    for j in range(len(S)):
        if S[j] == 'o':
            score += A[j]
    score += i + 1  # Add the bonus score
    return score

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    S = [sys.stdin.readline().strip() for _ in range(N)]

    for i in range(N):
        # Calculate the current total score of player i
        current_score = calculate_score(A, S[i], i)

        # Calculate the maximum total score of other players
        max_other_score = 0
        for j in range(N):
            if i != j:
                max_other_score = max(max_other_score, calculate_score(A, S[j], j))

        # Initialize the count of problems to solve
        count = 0

        # Sort the unsolved problems by their scores in descending order
        unsolved_problems = sorted([(A[j], j) for j in range(M) if S[i][j] == 'x'], reverse=True)

        # Try to solve the problems with the highest scores first
        while current_score <= max_other_score:
            if not unsolved_problems:
                break
            score, index = unsolved_problems.pop(0)
            current_score += score
            count += 1

        print(count)

if __name__ == "__main__":
    main()