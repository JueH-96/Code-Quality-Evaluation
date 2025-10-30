# YOUR CODE HERE

def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    S = [input() for _ in range(N)]

    scores = [0]*N
    for i in range(M):
        if S[i] == 'o':
            scores[i%N] += A[i]

    for i in range(N):
        if scores[i] < sum(scores):
            print(i+1)

if __name__ == "__main__":
    solve()