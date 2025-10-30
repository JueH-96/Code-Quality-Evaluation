# YOUR CODE HERE
n, m = map(int, input().split())
scores = list(map(int, input().split()))
players = []
for i in range(n):
    s = input().strip()
    players.append(s)

# Calculate current total scores for each player
current_scores = []
for i in range(n):
    total = 0
    for j in range(m):
        if players[i][j] == 'o':
            total += scores[j]
    total += (i + 1)  # bonus points
    current_scores.append(total)

# For each player, find minimum problems to solve
for i in range(n):
    # Find the maximum score among other players
    max_other_score = max(current_scores[j] for j in range(n) if j != i)
    
    # Get unsolved problems for player i and sort by score (descending)
    unsolved_problems = []
    for j in range(m):
        if players[i][j] == 'x':
            unsolved_problems.append(scores[j])
    
    unsolved_problems.sort(reverse=True)
    
    # Find minimum number of problems to solve
    current_score = current_scores[i]
    problems_solved = 0
    
    for score in unsolved_problems:
        if current_score > max_other_score:
            break
        current_score += score
        problems_solved += 1
    
    print(problems_solved)