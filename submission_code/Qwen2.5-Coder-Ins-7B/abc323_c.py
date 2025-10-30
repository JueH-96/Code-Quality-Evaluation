# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+M]))
S = data[2+M:]

def solve(N, M, A, S):
    results = []
    for i in range(N):
        solved = [j for j in range(M) if S[i][j] == 'o']
        unsolved = [j for j in range(M) if S[i][j] == 'x']
        current_score = sum(A[j] for j in solved) + i + 1
        max_score = current_score
        for j in unsolved:
            max_score = max(max_score, current_score + A[j])
        for j in unsolved:
            if current_score + A[j] > max_score:
                results.append(0)
                break
            else:
                max_score -= A[j]
        else:
            results.append(len(unsolved))
    return results

results = solve(N, M, A, S)
for result in results:
    print(result)