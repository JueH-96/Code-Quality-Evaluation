# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
S = data[2:]

max_days = 0

for i in range(D):
    free_days = 0
    for j in range(N):
        if S[j][i] == 'o':
            free_days += 1
        else:
            free_days = 0
        if free_days > max_days:
            max_days = free_days

print(max_days)