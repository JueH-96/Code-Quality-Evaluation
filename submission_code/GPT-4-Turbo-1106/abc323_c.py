# Read the number of players and problems
N, M = map(int, input().split())

# Read the scores for each problem
A = list(map(int, input().split()))

# Read the solved problems for each player
S = [input() for _ in range(N)]

# Calculate the total score for each player
total_scores = []
for i in range(N):
    score = sum(A[j] for j in range(M) if S[i][j] == 'o') + (i + 1)
    total_scores.append(score)

# For each player, calculate the minimum number of problems to solve to exceed all other players
for i in range(N):
    # The score needed to exceed all other players
    score_needed = max(total_scores[:i] + total_scores[i+1:]) + 1
    # The number of problems that need to be solved
    problems_to_solve = 0
    # Sort the unsolved problems by their score in descending order
    unsolved_scores = sorted([A[j] for j in range(M) if S[i][j] == 'x'], reverse=True)
    # Solve problems until the score exceeds the required score
    for unsolved_score in unsolved_scores:
        if total_scores[i] >= score_needed:
            break
        total_scores[i] += unsolved_score
        problems_to_solve += 1
    print(problems_to_solve)