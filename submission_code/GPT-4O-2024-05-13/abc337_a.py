# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
takahashi_total = 0
aoki_total = 0

for i in range(N):
    X = int(data[2 * i + 1])
    Y = int(data[2 * i + 2])
    takahashi_total += X
    aoki_total += Y

if takahashi_total > aoki_total:
    print("Takahashi")
elif aoki_total > takahashi_total:
    print("Aoki")
else:
    print("Draw")