# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def solve_adventure(N, events):
    potions = defaultdict(deque)
    used_potions = defaultdict(int)
    actions = [0] * N
    max_potions = 0
    current_potions = 0

    for i, (t, x) in enumerate(events):
        if t == 1:
            potions[x].append(i)
            actions[i] = 1
            current_potions += 1
            max_potions = max(max_potions, current_potions)
        elif t == 2:
            if not potions[x]:
                return -1
            else:
                used_potions[x] += 1
                while potions[x] and used_potions[x] > 0:
                    idx = potions[x].popleft()
                    actions[idx] = 0
                    used_potions[x] -= 1
                    current_potions -= 1

    return max_potions, actions

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    events = [(int(data[2 * i + 1]), int(data[2 * i + 2])) for i in range(N)]
    
    result = solve_adventure(N, events)
    if result == -1:
        print(-1)
    else:
        max_potions, actions = result
        print(max_potions)
        print(' '.join(str(action) for action in actions if events[action][0] == 1))

if __name__ == "__main__":
    main()