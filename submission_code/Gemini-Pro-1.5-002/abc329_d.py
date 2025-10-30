# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    for i in range(1, m + 1):
        counts = [0] * n
        for j in range(i):
            counts[a[j]-1] += 1
        
        max_votes = -1
        winner = -1
        for j in range(n):
            if counts[j] > max_votes:
                max_votes = counts[j]
                winner = j + 1
            elif counts[j] == max_votes and j + 1 < winner:
                winner = j + 1
        print(winner)

solve()