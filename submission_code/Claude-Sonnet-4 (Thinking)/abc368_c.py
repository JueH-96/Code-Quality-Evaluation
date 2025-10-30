def turns_to_kill(health, start_turn):
    if health <= 0:
        return 0
    
    # Find the damage pattern for the first 3 attacks
    damages = []
    for i in range(3):
        if (start_turn + i) % 3 == 0:
            damages.append(3)
        else:
            damages.append(1)
    
    # Every 3 attacks deal 5 damage
    full_cycles = health // 5
    turns = full_cycles * 3
    health -= full_cycles * 5
    
    # Handle the remainder
    for i in range(3):
        if health <= 0:
            break
        health -= damages[i]
        turns += 1
    
    return turns

n = int(input())
healths = list(map(int, input().split()))

total_turns = 0
for health in healths:
    turns_needed = turns_to_kill(health, total_turns + 1)
    total_turns += turns_needed

print(total_turns)