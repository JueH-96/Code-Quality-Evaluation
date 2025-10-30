# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
index = 1
bets = []
for _ in range(N):
    C_i = int(input[index])
    A_i = list(map(int, input[index + 1:index + 1 + C_i]))
    bets.append((C_i, A_i))
    index += 1 + C_i
X = int(input[index])

min_bets = float('inf')
people_with_min_bets = []

for i in range(N):
    if X in bets[i][1]:
        if bets[i][0] < min_bets:
            min_bets = bets[i][0]
            people_with_min_bets = [i + 1]
        elif bets[i][0] == min_bets:
            people_with_min_bets.append(i + 1)

print(len(people_with_min_bets))
print(" ".join(map(str, sorted(people_with_min_bets))))