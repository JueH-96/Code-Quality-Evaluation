import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    votes = [0]*N
    for i in range(N):
        votes[i] = A[i] - (sum(votes) - A[i])

    votes_left = K - sum(A)
    if votes_left < 0:
        print(-1)
        return

    votes_needed = [0]*N
    for i in range(N):
        if votes[i] < M:
            votes_needed[i] = M - votes[i]

    votes_needed_sum = sum(votes_needed)
    if votes_left < votes_needed_sum:
        print(-1)
        return

    votes_needed_sum -= votes_left
    votes_needed = [v + (votes_needed_sum // votes_needed_sum) for v in votes_needed]

    print(*votes_needed, sep=' ')

if __name__ == "__main__":
    solve()