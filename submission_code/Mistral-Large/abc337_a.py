import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
takahashi_score = 0
aoki_score = 0

for i in range(N):
    X = int(data[2 * i + 1])
    Y = int(data[2 * i + 2])
    takahashi_score += X
    aoki_score += Y

if takahashi_score > aoki_score:
    print("Takahashi")
elif takahashi_score < aoki_score:
    print("Aoki")
else:
    print("Draw")