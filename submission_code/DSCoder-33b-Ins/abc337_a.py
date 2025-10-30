# YOUR CODE HERE
N = int(input())
takahashi_score = 0
aoki_score = 0

for _ in range(N):
    X, Y = map(int, input().split())
    takahashi_score += X
    aoki_score += Y

if takahashi_score > aoki_score:
    print("Takahashi")
elif takahashi_score < aoki_score:
    print("Aoki")
else:
    print("Draw")