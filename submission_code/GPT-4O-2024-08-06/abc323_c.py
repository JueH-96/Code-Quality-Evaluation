# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    S = data[2+M:2+M+N]
    
    # Calculate the current scores for each player
    current_scores = []
    for i in range(N):
        score = sum(A[j] for j in range(M) if S[i][j] == 'o') + (i + 1)
        current_scores.append(score)
    
    # For each player, determine the minimum number of unsolved problems they need to solve
    results = []
    for i in range(N):
        # Calculate the score of player i
        player_score = current_scores[i]
        
        # Calculate the maximum score of other players
        max_other_score = max(current_scores[j] for j in range(N) if j != i)
        
        # Calculate the unsolved problems and their scores
        unsolved_problems = [(A[j], j) for j in range(M) if S[i][j] == 'x']
        unsolved_problems.sort(reverse=True, key=lambda x: x[0])  # Sort by score descending
        
        # Determine the minimum number of problems to solve
        additional_score_needed = max_other_score - player_score + 1
        solved_count = 0
        for score, _ in unsolved_problems:
            if additional_score_needed <= 0:
                break
            additional_score_needed -= score
            solved_count += 1
        
        results.append(solved_count)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()