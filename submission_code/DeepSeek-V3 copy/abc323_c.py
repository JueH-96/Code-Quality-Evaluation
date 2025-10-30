# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    S = data[2+M:2+M+N]
    
    # Calculate current scores for all players
    current_scores = []
    for i in range(N):
        score = 0
        for j in range(M):
            if S[i][j] == 'o':
                score += A[j]
        score += (i+1)
        current_scores.append(score)
    
    # For each player, determine the minimum number of problems to solve
    for i in range(N):
        # Find the maximum score among other players
        max_other = max([current_scores[j] for j in range(N) if j != i])
        
        # Calculate the required additional score
        required = max_other - current_scores[i] + 1
        
        if required <= 0:
            print(0)
            continue
        
        # Find the list of unsolved problems for player i
        unsolved = [A[j] for j in range(M) if S[i][j] == 'x']
        unsolved.sort(reverse=True)
        
        # Calculate the minimum number of problems to solve
        count = 0
        total = 0
        for a in unsolved:
            total += a
            count += 1
            if total >= required:
                break
        print(count)

if __name__ == "__main__":
    main()