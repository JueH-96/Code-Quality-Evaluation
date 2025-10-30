# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
takahashi_score = 0
aoki_score = 0

for i in range(N):
    takahashi_score += int(data[2*i+1])
    aoki_score += int(data[2*i+2])

if takahashi_score > aoki_score:
    print("Takahashi")
elif takahashi_score < aoki_score:
    print("Aoki")
else:
    print("Draw")