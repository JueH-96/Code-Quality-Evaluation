# YOUR CODE HERE
import sys
from collections import defaultdict

# Use sys.stdin.readline for faster input reading
# This is a common practice in competitive programming to handle large inputs efficiently.
input = sys.stdin.readline 

def solve():
    # Read the number of events
    N = int(input())
    
    # Store events as a list of lists. Each inner list is [type, value].
    # The index in this 'events' list corresponds to the event order (0 to N-1).
    events = []
    for _ in range(N):
        events.append(list(map(int, input().split())))

    # required[x]: Tracks the net requirement of potions of type x needed for monsters
    # encountered later in the sequence (during the reverse pass). A positive value indicates
    # a deficit of potions found later that must be satisfied by potions found earlier.
    required = defaultdict(int)
    
    # pickup[i]: Stores the decision (0 for discard, 1 for pick up) for the i-th event 
    # if it's a type 1 event (potion find). Initialize all to 0 (default action is discard).
    pickup = [0] * N 
    
    # --- Reverse Pass ---
    # Iterate through events from the last (index N-1) down to the first (index 0).
    # This pass determines greedily which potions MUST be picked up. The logic is based on
    # satisfying each monster requirement with the latest possible potion find event.
    for i in range(N - 1, -1, -1):
        t_i, x_i = events[i]
        
        if t_i == 2: # Monster encounter
            # Increment the requirement count for potion type x_i.
            # This monster requires a potion of type x_i available at or before event i.
            required[x_i] += 1 
        else: # t_i == 1, Potion find
            # Check if this potion is needed to satisfy any future monster requirement
            # identified so far in the reverse pass.
            if required[x_i] > 0:
                # If yes, mark this potion to be picked up.
                pickup[i] = 1     
                # Decrease the requirement count, as this potion satisfies one need.
                required[x_i] -= 1 
            # If required[x_i] is 0, it means all monsters of type x_i encountered from event i+1 onwards
            # are already covered by potions found between event i+1 and N.
            # Therefore, this potion at event i is not needed for future encounters based on the
            # 'use latest available potion' principle. We leave pickup[i] as 0 (discard)
            # to minimize the number of potions held at any time.

    # --- Forward Pass Simulation ---
    # Simulate the adventure chronologically from event 0 to N-1 using the pickup decisions 
    # determined in the reverse pass.
    # Calculate the maximum number of potions held simultaneously (K_min).
    
    # potion_count[x]: Stores the current count of potions of type x held by Takahashi.
    potion_count = defaultdict(int) 
    # total_potions: Tracks the current total number of potions held across all types.
    total_potions = 0              
    # max_potions_held: Stores the maximum value 'total_potions' reaches during the simulation.
    # This will be the final K_min value.
    max_potions_held = 0           
    
    # action_sequence: A list to store the actions (0 or 1) for type 1 events, in the order
    # they appear in the input.
    action_sequence = [] 
    
    # possible: A flag indicating whether Takahashi can survive the adventure using this strategy.
    # Initialize to True, assuming survival is possible until proven otherwise.
    possible = True 
    
    # Iterate through events from the first (index 0) to the last (index N-1).
    for i in range(N):
        t_i, x_i = events[i]
        
        if t_i == 1: # Potion find event
            # Check the decision made during the reverse pass for this potion.
            if pickup[i] == 1: 
                # If decided to pick up:
                potion_count[x_i] += 1 # Increment count for this potion type.
                total_potions += 1    # Increment total potions count.
            
            # Record the action taken (0 or 1) for this type 1 event.
            # This builds the sequence required for the output.
            action_sequence.append(pickup[i]) 

        else: # t_i == 2, Monster encounter event
            # Check if Takahashi currently possesses a potion of the required type x_i.
            if potion_count[x_i] > 0:
                # If yes, use one potion:
                potion_count[x_i] -= 1 # Decrement count for this potion type.
                total_potions -= 1    # Decrement total potions count.
            else:
                # If no potion of type x_i is available, Takahashi cannot defeat the monster.
                # The strategy derived from the reverse pass has failed.
                # Since this strategy aimed to pick up the minimum necessary set of potions
                # to survive based on a greedy choice ('latest possible potion'), its failure
                # implies that survival is impossible under any strategy.
                possible = False
                break # Stop the simulation as the outcome is determined.
        
        # After each event, update the maximum number of potions held if the current total is greater.
        max_potions_held = max(max_potions_held, total_potions)

    # --- Output ---
    if not possible:
        # If the simulation determined that survival is impossible.
        print("-1")
    else:
        # If survival is possible:
        # Print K_min (the maximum number of potions held at any point).
        print(max_potions_held)
        # Print the sequence of actions for type 1 events, separated by spaces.
        # The `*` operator unpacks the list elements for the print function.
        print(*(action_sequence)) 

# Execute the main solver function.
solve()