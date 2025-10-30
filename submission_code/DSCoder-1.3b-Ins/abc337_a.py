# YOUR CODE HERE
N = int(input())
scores = []
for _ in range(N):
    X, Y = map(int, input().split())
    scores.append(X + Y)

if scores.count(max(scores)) == 1:
    print("Takahashi")
elif scores.count(min(scores)) == 1:
    print("Aoki")
else:
    print("Draw")