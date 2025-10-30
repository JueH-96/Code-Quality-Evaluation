def attacks_needed(T, health):
    # Full cycles of 3 attacks
    cycles = health // 5
    attacks_so_far = 3 * cycles
    remaining_health = health - 5 * cycles
    
    if remaining_health == 0:
        return attacks_so_far
    
    remainder = (T + attacks_so_far) % 3
    
    if remainder == 0:  # Pattern: 1, 1, 3
        if remaining_health == 1:
            return attacks_so_far + 1
        elif remaining_health == 2:
            return attacks_so_far + 2
        elif remaining_health == 3:
            return attacks_so_far + 3
        else:  # remaining_health == 4
            return attacks_so_far + 3
    elif remainder == 1:  # Pattern: 1, 3, 1
        if remaining_health == 1:
            return attacks_so_far + 1
        elif remaining_health == 2:
            return attacks_so_far + 2
        elif remaining_health == 3:
            return attacks_so_far + 2
        else:  # remaining_health == 4
            return attacks_so_far + 3
    else:  # remainder == 2, Pattern: 3, 1, 1
        if remaining_health == 1:
            return attacks_so_far + 1
        elif remaining_health == 2:
            return attacks_so_far + 1
        elif remaining_health == 3:
            return attacks_so_far + 1
        else:  # remaining_health == 4
            return attacks_so_far + 2

def minimum_attacks(N, healths):
    T = 0  # Initialize T to 0
    
    for i in range(N):
        attacks = attacks_needed(T, healths[i])
        T += attacks
    
    return T

# Read inputs
N = int(input())
healths = list(map(int, input().split()))

# Compute and print the result
print(minimum_attacks(N, healths))