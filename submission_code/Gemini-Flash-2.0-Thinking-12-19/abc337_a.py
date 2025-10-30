N = int(input())
takahashi_score = 0
aoki_score = 0
for _ in range(N):
    scores = input().split()
    x_i = int(scores[0])
    y_i = int(scores[1])
    takahashi_score += x_i
    aoki_score += y_i

if takahashi_score > aoki_score:
    print("Takahashi")
elif aoki_score > takahashi_score:
    print("Aoki")
else:
    print("Draw")