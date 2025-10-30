# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
index = 1
bets = []
for i in range(N):
    C_i = int(data[index])
    index += 1
    A_i = list(map(int, data[index:index + C_i]))
    index += C_i
    bets.append((C_i, A_i))

X = int(data[index])

# Find all people who have bet on X
people_bet_on_X = []
for i in range(N):
    C_i, A_i = bets[i]
    if X in A_i:
        people_bet_on_X.append((i + 1, C_i))

# Find the minimum number of bets among those who bet on X
if people_bet_on_X:
    min_bets = min(person[1] for person in people_bet_on_X)
    result = [person[0] for person in people_bet_on_X if person[1] == min_bets]
    result.sort()
    print(len(result))
    print(' '.join(map(str, result)))
else:
    print(0)