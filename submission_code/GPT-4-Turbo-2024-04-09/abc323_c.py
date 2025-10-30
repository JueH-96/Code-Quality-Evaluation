import sys
input = sys.stdin.read

def solve():
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    scores = [int(data[i]) for i in range(idx, idx + M)]
    idx += M
    
    players = []
    for i in range(N):
        players.append(data[idx])
        idx += 1
    
    # Calculate current scores
    current_scores = []
    for i in range(N):
        score = i + 1  # bonus score
        for j in range(M):
            if players[i][j] == 'o':
                score += scores[j]
        current_scores.append(score)
    
    results = []
    
    for i in range(N):
        max_other_score = max(current_scores[:i] + current_scores[i+1:])
        needed_score = max_other_score + 1
        current_score = current_scores[i]
        
        if current_score > max_other_score:
            results.append(0)
            continue
        
        # Calculate the minimum problems to solve to exceed max_other_score
        unsolved_problems = []
        for j in range(M):
            if players[i][j] == 'x':
                unsolved_problems.append(scores[j])
        
        # Sort unsolved problems by score descending
        unsolved_problems.sort(reverse=True)
        
        sum_score = current_score
        count = 0
        for score in unsolved_problems:
            sum_score += score
            count += 1
            if sum_score > max_other_score:
                results.append(count)
                break
    
    for result in results:
        print(result)