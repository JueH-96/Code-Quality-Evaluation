def solve():
    n = int(input())
    h = list(map(int, input().split()))
    current_healths = list(h)
    total_turns = 0
    
    for i in range(n):
        target_health = current_healths[i]
        if target_health <= 0:
            continue
            
        start_turn = total_turns + 1
        low = 1
        high = 3 * target_health
        turns_to_defeat = 0
        
        while low <= high:
            mid_turns = (low + high) // 2
            damage_turns_3_count = (start_turn + mid_turns - 1) // 3 - (start_turn - 1) // 3
            damage = damage_turns_3_count * 3 + (mid_turns - damage_turns_3_count) * 1
            if damage >= target_health:
                turns_to_defeat = mid_turns
                high = mid_turns - 1
            else:
                low = mid_turns + 1
                
        total_turns += turns_to_defeat
        damage_turns_3_count = (start_turn + turns_to_defeat - 1) // 3 - (start_turn - 1) // 3
        damage_dealt = damage_turns_3_count * 3 + (turns_to_defeat - damage_turns_3_count) * 1
        current_healths[i] -= damage_dealt
        
    print(total_turns)

if __name__ == '__main__':
    solve()