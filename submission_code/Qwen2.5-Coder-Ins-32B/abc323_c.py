# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    index += 2
    A = list(map(int, input[index:index + M]))
    index += M
    S = input[index:index + N]
    
    current_scores = [0] * N
    for i in range(N):
        for j in range(M):
            if S[i][j] == 'o':
                current_scores[i] += A[j]
        current_scores[i] += i + 1
    
    results = []
    for i in range(N):
        unsolved_scores = []
        for j in range(M):
            if S[i][j] == 'x':
                unsolved_scores.append(A[j])
        unsolved_scores.sort(reverse=True)
        
        target_score = max(current_scores) + 1
        current_player_score = current_scores[i] - (i + 1)
        problems_solved = 0
        
        for score in unsolved_scores:
            current_player_score += score
            problems_solved += 1
            if current_player_score >= target_score:
                break
        
        results.append(problems_solved)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()