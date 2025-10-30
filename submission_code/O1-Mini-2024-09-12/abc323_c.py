# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    A = list(map(int, data[idx:idx+M]))
    idx += M
    S = []
    for _ in range(N):
        S.append(data[idx])
        idx +=1
    # Compute current scores
    scores = []
    for i in range(N):
        total = 0
        for j in range(M):
            if S[i][j] == 'o':
                total += A[j]
        total += (i+1)  # bonus
        scores.append(total)
    # For each player
    for i in range(N):
        # Compute max_other_score
        max_other = max(scores[:i] + scores[i+1:])
        current = scores[i]
        if current > max_other:
            print(0)
            continue
        difference = max_other - current
        # Get unsolved problems
        unsolved = []
        for j in range(M):
            if S[i][j] == 'x':
                unsolved.append(A[j])
        unsolved.sort(reverse=True)
        prefix = 0
        k = 0
        while k < len(unsolved):
            prefix += unsolved[k]
            if prefix > difference:
                print(k+1)
                break
            k +=1
    return

if __name__ == "__main__":
    main()