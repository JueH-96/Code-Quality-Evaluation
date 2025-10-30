import sys

def solve():
    # Read N, the number of enemies
    N = int(sys.stdin.readline())
    
    # Read the healths of N enemies into a list
    H = list(map(int, sys.stdin.readline().split()))

    # Initialize total_T to 0. This variable will accumulate the total turns.
    total_T = 0

    # Iterate through each enemy's health
    for h_i in H:
        h = h_i  # Current enemy's health

        # Phase 1: Efficiently reduce large health chunks
        # We know that every 3 turns, 5 damage (1 + 1 + 3) is dealt.
        # Calculate how many full 5-damage cycles can be applied to the current enemy.
        num_full_cycles = h // 5
        
        # Add the turns taken for these full cycles to total_T.
        total_T += num_full_cycles * 3
        
        # Reduce the current enemy's health by the damage dealt in these full cycles.
        # This leaves h with a value between 0 and 4 (h % 5).
        h -= num_full_cycles * 5 

        # Phase 2: Deal with the remaining small amount of health (0, 1, 2, 3, or 4)
        # This part requires turn-by-turn simulation because the exact damage (1 or 3)
        # depends on the current total_T's remainder when divided by 3.
        # Since h is very small here, this loop will run at most 3-4 times, making it very fast.
        while h > 0:
            # Increment T for the current turn. This is the first action of a turn.
            total_T += 1
            
            # Determine the damage dealt on this specific turn.
            # If the current T is a multiple of 3, damage is 3; otherwise, it's 1.
            damage = 3 if total_T % 3 == 0 else 1
            
            # Apply the damage to the enemy's health.
            h -= damage
            
    # Print the final total number of turns required to defeat all enemies.
    print(total_T)

# Call the solve function to run the program.
solve()