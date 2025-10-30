# YOUR CODE HERE
from collections import defaultdict, deque

def adventure(N, events):
    potions = defaultdict(int)
    actions = []
    max_potions = 0
    current_potions = 0
    
    for t, x in events:
        if t == 1:
            potions[x] += 1
            actions.append((t, x, 1))
            current_potions += 1
            max_potions = max(max_potions, current_potions)
        elif t == 2:
            if potions[x] > 0:
                potions[x] -= 1
                current_potions -= 1
                actions.append((t, x, 0))
            else:
                return -1, []
    
    result_actions = []
    for action in actions:
        if action[0] == 1:
            result_actions.append(action[2])
    
    return max_potions, result_actions

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
events = []
for i in range(1, len(data), 2):
    t = int(data[i])
    x = int(data[i+1])
    events.append((t, x))

# Solve the problem
max_potions, result_actions = adventure(N, events)

# Print the result
if max_potions == -1:
    print(-1)
else:
    print(max_potions)
    print(" ".join(map(str, result_actions)))