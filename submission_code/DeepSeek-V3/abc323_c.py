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
        score += (i+1)  # bonus
        current_scores.append(score)
    
    # For each player, determine the minimum number of problems to solve
    for i in range(N):
        # Get the current score of player i
        my_score = current_scores[i]
        # Find the maximum score among other players
        max_other = max([current_scores[j] for j in range(N) if j != i])
        # If already ahead, no need to solve any
        if my_score > max_other:
            print(0)
            continue
        # Otherwise, find the minimum number of problems to solve
        # Get the list of unsolved problems for player i
        unsolved = [j for j in range(M) if S[i][j] == 'x']
        # Sort the unsolved problems in descending order of A[j]
        unsolved.sort(key=lambda x: A[x], reverse=True)
        # Calculate the required additional score
        required = max_other - my_score + 1
        # Sum the largest A[j] until the required is met
        count = 0
        for j in unsolved:
            required -= A[j]
            count += 1
            if required <= 0:
                break
        print(count)

if __name__ == "__main__":
    main()