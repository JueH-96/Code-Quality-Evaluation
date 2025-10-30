# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
cups = []

for i in range(N):
    F = int(data[2*i+1])
    S = int(data[2*i+2])
    cups.append((F, S))

max_satisfaction = 0

for i in range(N):
    for j in range(i+1, N):
        F1, S1 = cups[i]
        F2, S2 = cups[j]
        if F1 != F2:
            satisfaction = S1 + S2
        else:
            satisfaction = S1 + S2 // 2
        if satisfaction > max_satisfaction:
            max_satisfaction = satisfaction

print(max_satisfaction)