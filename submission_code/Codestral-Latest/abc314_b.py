# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1

bets = []
for i in range(N):
    C_i = int(data[index])
    index += 1
    A_i = set(map(int, data[index:index + C_i]))
    index += C_i
    bets.append(A_i)

X = int(data[index])

# Find the minimum number of bets for people who bet on X
min_bets = float('inf')
for i in range(N):
    if X in bets[i]:
        min_bets = min(min_bets, len(bets[i]))

# Find all people who bet on X with the minimum number of bets
result = []
for i in range(N):
    if X in bets[i] and len(bets[i]) == min_bets:
        result.append(i + 1)

# Print the result
print(len(result))
if result:
    print(' '.join(map(str, result)))