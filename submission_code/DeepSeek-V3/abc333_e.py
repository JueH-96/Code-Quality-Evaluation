# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    events = []
    for _ in range(N):
        t, x = map(int, sys.stdin.readline().split())
        events.append((t, x))
    
    # First, check if it's possible to defeat all monsters
    # We need to ensure that for every monster, there is at least one potion of its type available before it
    # We can do this by counting the number of potions and monsters for each type
    potion_counts = defaultdict(int)
    monster_counts = defaultdict(int)
    for t, x in events:
        if t == 1:
            potion_counts[x] += 1
        else:
            monster_counts[x] += 1
    for x in monster_counts:
        if potion_counts[x] < monster_counts[x]:
            print(-1)
            return
    
    # Now, we need to find the minimal K_min
    # K_min is the minimal maximum number of potions Takahashi has at any point
    # To minimize K_min, we need to pick up potions only when necessary
    # We can simulate the process and keep track of the current number of potions
    # We will use a greedy approach: pick up a potion only when it is needed to defeat a monster
    
    # We will process the events in reverse order to determine which potions to pick up
    # We will use a stack to keep track of the potions we need to pick up
    stack = []
    # We will also keep a dictionary to count the number of potions needed for each type
    needed = defaultdict(int)
    for t, x in reversed(events):
        if t == 2:
            needed[x] += 1
        else:
            if needed[x] > 0:
                stack.append((x, 1))
                needed[x] -= 1
            else:
                stack.append((x, 0))
    
    # Now, we will process the events in the original order and simulate the potion counts
    current_potions = defaultdict(int)
    max_potions = 0
    actions = []
    for t, x in events:
        if t == 1:
            # Check if we need to pick up this potion
            if stack[-1][1] == 1:
                current_potions[x] += 1
                actions.append(1)
            else:
                actions.append(0)
            stack.pop()
        else:
            if current_potions[x] > 0:
                current_potions[x] -= 1
            else:
                # This should not happen as we have already checked the counts
                pass
        # Update max_potions
        total = sum(current_potions.values())
        if total > max_potions:
            max_potions = total
    
    print(max_potions)
    print(' '.join(map(str, actions)))

if __name__ == "__main__":
    main()