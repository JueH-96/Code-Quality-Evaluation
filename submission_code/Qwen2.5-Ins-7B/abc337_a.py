# YOUR CODE HERE
n = int(input())
takahashi_score = 0
aoki_score = 0

for i in range(n):
    x, y = map(int, input().split())
    takahashi_score += x
    aoki_score += y

if takahashi_score > aoki_score:
    print("Takahashi")
elif takahashi_score < aoki_score:
    print("Aoki")
else:
    print("Draw")