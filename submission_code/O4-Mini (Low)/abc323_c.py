import sys

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    S = [input().strip() for _ in range(N)]
    
    # Compute current scores
    curr = []
    for i in range(N):
        score = (i+1)  # bonus
        for j in range(M):
            if S[i][j] == 'o':
                score += A[j]
        curr.append(score)
    
    # For each player, find how many more solves to exceed others
    for i in range(N):
        # maximum score among other players
        max_other = max(curr[:i] + curr[i+1:])
        my_score = curr[i]
        if my_score > max_other:
            print(0)
            continue
        
        # list of unsolved problem scores, descending
        unsolved = []
        for j in range(M):
            if S[i][j] == 'x':
                unsolved.append(A[j])
        unsolved.sort(reverse=True)
        
        # pick largest until exceeding
        added = 0
        count = 0
        for v in unsolved:
            added += v
            count += 1
            if my_score + added > max_other:
                print(count)
                break

if __name__ == "__main__":
    main()