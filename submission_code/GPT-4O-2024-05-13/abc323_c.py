# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    S = data[2+M:]
    
    current_scores = []
    for i in range(N):
        score = sum(A[j] for j in range(M) if S[i][j] == 'o') + (i + 1)
        current_scores.append(score)
    
    results = []
    for i in range(N):
        target_score = current_scores[i]
        other_scores = current_scores[:i] + current_scores[i+1:]
        max_other_score = max(other_scores)
        
        unsolved_problems = [A[j] for j in range(M) if S[i][j] == 'x']
        unsolved_problems.sort(reverse=True)
        
        additional_score_needed = max_other_score - target_score + 1
        additional_score = 0
        problems_solved = 0
        
        for score in unsolved_problems:
            additional_score += score
            problems_solved += 1
            if additional_score >= additional_score_needed:
                break
        
        results.append(problems_solved)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()