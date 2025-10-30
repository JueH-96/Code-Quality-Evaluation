def solve():
    n, m = map(int, input().split())
    problem_scores = list(map(int, input().split()))
    solution_strings = [input() for _ in range(n)]
    
    current_scores = []
    for i in range(n):
        player_score = 0
        for j in range(m):
            if solution_strings[i][j] == 'o':
                player_score += problem_scores[j]
        current_scores.append(player_score + (i + 1))
        
    results = []
    for i in range(n):
        max_other_score = 0
        for j in range(n):
            if i != j:
                max_other_score = max(max_other_score, current_scores[j])
                
        if current_scores[i] > max_other_score:
            results.append(0)
        else:
            unsolved_problems_scores = []
            unsolved_problem_indices = []
            for j in range(m):
                if solution_strings[i][j] == 'x':
                    unsolved_problems_scores.append(problem_scores[j])
                    
            unsolved_problems_scores.sort(reverse=True)
            needed_problems_count = 0
            current_player_score = current_scores[i]
            solved_problem_index = 0
            while current_player_score <= max_other_score:
                if solved_problem_index < len(unsolved_problems_scores):
                    current_player_score += unsolved_problems_scores[solved_problem_index]
                    needed_problems_count += 1
                    solved_problem_index += 1
                else:
                    # This case should not happen based on problem description, but for robustness.
                    break 
            results.append(needed_problems_count)
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()