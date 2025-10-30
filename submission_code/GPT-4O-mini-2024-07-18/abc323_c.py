def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Read scores for each problem
    scores = list(map(int, data[1].split()))
    
    # Read the solved problems for each player
    solved = data[2:2 + N]
    
    # Calculate total scores for each player
    total_scores = []
    for i in range(N):
        score = sum(scores[j] for j in range(M) if solved[i][j] == 'o')
        total_scores.append(score + (i + 1))  # Add bonus score
    
    # Prepare to store results
    results = []
    
    # Calculate the minimum number of problems to solve for each player
    for i in range(N):
        current_score = total_scores[i]
        needed_score = max(total_scores[j] for j in range(N) if j != i) + 1
        
        # Calculate how many more problems need to be solved
        remaining_scores = [scores[j] for j in range(M) if solved[i][j] == 'x']
        remaining_scores.sort(reverse=True)  # Sort in descending order
        
        additional_score = 0
        problems_solved = 0
        
        for score in remaining_scores:
            additional_score += score
            problems_solved += 1
            if current_score + additional_score >= needed_score:
                break
        
        results.append(problems_solved)
    
    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()