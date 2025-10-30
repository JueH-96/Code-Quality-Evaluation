def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    events = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]
    
    # Dictionary to count required potions for each type
    required_potions = {}
    # Dictionary to count found potions for each type
    found_potions = {}
    
    # First pass to determine the minimum required potions for each type
    for t, x in events:
        if t == 1:
            if x in found_potions:
                found_potions[x] += 1
            else:
                found_potions[x] = 1
        elif t == 2:
            if x in required_potions:
                required_potions[x] += 1
            else:
                required_potions[x] = 1
    
    # Check if it's possible to defeat all monsters
    for monster_type, count in required_potions.items():
        if count > found_potions.get(monster_type, 0):
            print(-1)
            return
    
    # Second pass to determine the minimum K_min
    # Reset found potions
    found_potions = {}
    pick_decisions = []
    
    # Track the minimum K that allows defeating all monsters
    K_min = 0
    current_potions = {}
    
    for t, x in events:
        if t == 1:
            # Consider picking up this potion
            if x in current_potions:
                current_potions[x] += 1
            else:
                current_potions[x] = 1
            
            # Record the decision to pick up
            pick_decisions.append(1)
            
            # Update K_min if necessary
            K_min = max(K_min, sum(current_potions.values()))
        elif t == 2:
            # Use the potion to defeat the monster
            if current_potions[x] > 0:
                current_potions[x] -= 1
            else:
                # This should never happen because we checked feasibility before
                print(-1)
                return
    
    # Output the results
    print(K_min)
    # Only output decisions for t_i = 1
    print(' '.join(str(decision) for decision, (t, x) in zip(pick_decisions, events) if t == 1))

if __name__ == "__main__":
    main()