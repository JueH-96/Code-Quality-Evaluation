def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    s_list = [input() for _ in range(n)]
    
    current_scores = [0] * n
    for i in range(n):
        score = 0
        for j in range(m):
            if s_list[i][j] == 'o':
                score += a[j]
        score += (i + 1)
        current_scores[i] = score
        
    results = []
    for i in range(n):
        max_other_score = 0
        for j in range(n):
            if i != j:
                max_other_score = max(max_other_score, current_scores[j])
                
        if current_scores[i] > max_other_score:
            results.append(0)
            continue
            
        unsolved_problem_scores = []
        unsolved_problem_indices = []
        for j in range(m):
            if s_list[i][j] == 'x':
                unsolved_problem_scores.append(a[j])
                
        unsolved_problem_scores.sort(reverse=True)
        
        problems_to_solve_count = 0
        current_player_score = current_scores[i]
        
        for problem_score in unsolved_problem_scores:
            current_player_score += problem_score
            problems_to_solve_count += 1
            if current_player_score > max_other_score:
                results.append(problems_to_solve_count)
                break
                
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()