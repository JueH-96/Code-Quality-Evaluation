# YOUR CODE HERE
n = int(input())
wins = [0] * n
for i in range(n):
    s = input()
    for j in range(n):
        if s[j] == 'o':
            wins[i] += 1
rank = sorted(range(n), key=lambda k: (-wins[k], k))
for i in range(n):
    print(rank[i] + 1, end=' ')