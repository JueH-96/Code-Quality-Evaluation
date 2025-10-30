def main():
    n, m = map(int, input().split())
    scores = list(map(int, input().split()))
    solved_strings = [input().strip() for _ in range(n)]
    
    # Calculate the current scores of all players
    current_scores = []
    for i in range(n):
        score = i + 1  # Bonus points
        for j in range(m):
            if solved_strings[i][j] == 'o':
                score += scores[j]
        current_scores.append(score)
    
    for i in range(n):
        # Find the maximum score among other players
        max_other_score = max(current_scores[:i] + current_scores[i+1:])
        
        # If the current player's score is already higher, they don't need to solve any more problems
        if current_scores[i] > max_other_score:
            print(0)
            continue
        
        # Identify unsolved problems for the current player
        unsolved_scores = [scores[j] for j in range(m) if solved_strings[i][j] == 'x']
        
        # Sort unsolved problems by score in descending order
        unsolved_scores.sort(reverse=True)
        
        # Calculate the score gap that needs to be closed
        score_gap = max_other_score - current_scores[i] + 1  # +1 to exceed, not just equal
        
        # Count how many problems are needed to close the gap
        problems_needed = 0
        running_score = 0
        for score in unsolved_scores:
            running_score += score
            problems_needed += 1
            if running_score >= score_gap:
                break
        
        print(problems_needed)

if __name__ == "__main__":
    main()