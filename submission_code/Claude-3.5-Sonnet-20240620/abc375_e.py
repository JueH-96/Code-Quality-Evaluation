# YOUR CODE HERE
def can_equalize(teams, target):
    moves = 0
    for i in range(3):
        diff = teams[i] - target
        if diff > 0:
            moves += diff
    return moves // 2

N = int(input())
teams = [0, 0, 0]
people = []

for _ in range(N):
    a, b = map(int, input().split())
    teams[a-1] += b
    people.append((a-1, b))

total_strength = sum(teams)

if total_strength % 3 != 0:
    print(-1)
else:
    target = total_strength // 3
    max_strength = max(teams)
    
    if max_strength > target:
        moves = can_equalize(teams, target)
        print(moves)
    else:
        print(0)