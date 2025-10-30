import sys
from collections import defaultdict
import heapq

def solve():
    N = int(sys.stdin.readline())
    events = []
    for _ in range(N):
        t, x = map(int, sys.stdin.readline().split())
        events.append((t, x))

    # Pass 1: Determine which potions MUST be picked up and calculate K_min.
    
    # uncovered_monster_indices[x] stores a min-heap of event indices (i)
    # for monsters of type x that need a potion from a t=1 event occurring earlier than i (in forward time).
    uncovered_monster_indices = defaultdict(list) 
    
    # final_actions[i] will be 1 if potion at event i is picked up, 0 if discarded.
    # Only relevant for t=1 events. Initialized to 0 (discard by default).
    final_actions = [0] * N 

    # current_potions_held_count tracks the number of potions Takahashi needs to hold
    # or acquire at the current point (i, in forward time) to satisfy all future monster needs.
    current_potions_held_count = 0 
    
    # This will store the maximum value current_potions_held_count reaches, which is K_min.
    max_K_val = 0 

    # Iterate events in reverse order
    for i in range(N - 1, -1, -1):
        t, x = events[i]
        
        if t == 2:  # Monster encounter (type x)
            # This monster needs a potion of type `x`. Add its event index to the heap.
            heapq.heappush(uncovered_monster_indices[x], i)
            
            # This monster adds to the total number of potions Takahashi must conceptually hold.
            current_potions_held_count += 1
            
            # Update the maximum `current_potions_held_count` seen so far. This tracks K_min.
            max_K_val = max(max_K_val, current_potions_held_count)
            
        else:  # t == 1: Potion found (type x)
            # Check if this potion of type `x` is needed for any future monster.
            if len(uncovered_monster_indices[x]) > 0:
                # Yes, it's needed. We must pick it up.
                final_actions[i] = 1
                
                # This potion satisfies the need of the earliest monster of type x (smallest index).
                # We pop it from the heap as it's now 'covered'.
                heapq.heappop(uncovered_monster_indices[x])
                
                # `current_potions_held_count` does not change here. This potion fulfills an existing need,
                # so it doesn't add to the *net* number of potions actively held for future, unmet needs.
            else:
                # This potion is not needed for any future monster. Discard it to minimize K.
                final_actions[i] = 0
                
                # Since we discard it, it reduces the number of potions contributing to the count.
                current_potions_held_count -= 1
                
                # Impossibility Check: If `current_potions_held_count` drops below 0, it means
                # we've discarded a potion that was implicitly necessary to maintain the chain of satisfying
                # monster needs. This indicates Takahashi will be defeated.
                if current_potions_held_count < 0:
                    print("-1")
                    return

    # Final Impossibility Check for Pass 1:
    # If after the backward pass, any monster heap is not empty, it means there are monsters
    # that could not be satisfied by any preceding (in forward time) potion-finding event.
    # This would result in defeat.
    for x_type in uncovered_monster_indices:
        if len(uncovered_monster_indices[x_type]) > 0:
            print("-1")
            return
            
    # Pass 2: Simulate forward with the determined actions to explicitly verify possibility
    # and confirm the maximum K achieved (should match max_K_val from Pass 1).
    
    current_potions_inventory = defaultdict(int) # Counts of potions held by type
    total_potions_held = 0 # Total count of all potions held
    
    # This loop primarily serves as a robust verification.
    for i in range(N):
        t, x = events[i]
        
        if t == 1: # Potion found
            if final_actions[i] == 1: # Pick it up as decided in Pass 1
                current_potions_inventory[x] += 1
                total_potions_held += 1
            # else: discard, no change to potions held or total_potions_held
        else: # t == 2: Monster encounter
            if current_potions_inventory[x] > 0: # Check if potion of type x is available
                current_potions_inventory[x] -= 1
                total_potions_held -= 1
            else:
                # This means Takahashi doesn't have the required potion. Defeated.
                # This case should ideally not be reached if Pass 1's logic for impossibility is complete.
                print("-1")
                return

    # If all checks pass, Takahashi can defeat all monsters.
    # Print the minimum maximum number of potions (K_min).
    print(max_K_val)
    
    # Collect and print the decisions for t=1 events in their original order.
    output_actions = []
    for i in range(N):
        if events[i][0] == 1: # Check if it's a potion-finding event
            output_actions.append(str(final_actions[i]))
            
    print(" ".join(output_actions))

solve()