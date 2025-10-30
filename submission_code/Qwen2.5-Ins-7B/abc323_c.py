# NUMBER_OF_PLAYERS NUMBER_OF_PROBLEMS
NUMBER_OF_PLAYERS, NUMBER_OF_PROBLEMS = map(int, input().split())
# PROBLEM_SCORES
PROBLEM_SCORES = list(map(int, input().split()))
# PLAYERS_SOLUTIONS
PLAYERS_SOLUTIONS = [input() for _ in range(NUMBER_OF_PLAYERS)]

def solve(player_index):
    player_solved = PLAYERS_SOLUTIONS[player_index]
    player_score = sum(PROBLEM_SCORES[i] for i in range(NUMBER_OF_PROBLEMS) if player_solved[i] == 'o')
    player_score += player_index + 1
    problems_left = [i for i in range(NUMBER_OF_PROBLEMS) if PLAYERS_SOLUTIONS[player_index][i] == 'x']
    problems_left.sort(key=lambda x: PROBLEM_SCORES[x], reverse=True)
    for i, problem in enumerate(problems_left):
        player_score += PROBLEM_SCORES[problem]
        if all(player_score > sum(PROBLEM_SCORES[j] for j in range(NUMBER_OF_PROBLEMS) if PLAYERS_SOLUTIONS[p][j] == 'o') + p + 1 for p in range(NUMBER_OF_PLAYERS) if p != player_index):
            return i + 1
    return 0

for i in range(NUMBER_OF_PLAYERS):
    print(solve(i))