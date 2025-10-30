# YOUR CODE HERE
n, t = map(int, input().split())
scores = [0] * n
for _ in range(t):
    a, b = map(int, input().split())
    scores[a - 1] += b
    print(len(set(scores)))