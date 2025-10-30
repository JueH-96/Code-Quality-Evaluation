def solve():
    n = int(input())
    healths = list(map(int, input().split()))
    total_turns = 0
    for i in range(n):
        current_health = healths[i]
        k = current_health // 5
        turns = 3 * k
        current_health %= 5
        current_turn_start_for_additional = total_turns + turns
        additional_turns = 0
        if current_health == 1:
            additional_turns = 1
        elif current_health == 2:
            if current_turn_start_for_additional % 3 == 0:
                additional_turns = 1
            else:
                additional_turns = 2
        elif current_health == 3:
            if current_turn_start_for_additional % 3 == 0:
                additional_turns = 1
            elif current_turn_start_for_additional % 3 == 2:
                additional_turns = 2
            else:
                additional_turns = 3
        elif current_health == 4:
            if current_turn_start_for_additional % 3 == 0 or current_turn_start_for_additional % 3 == 2:
                additional_turns = 2
            else:
                additional_turns = 3
        else:
            additional_turns = 0
        turns_for_enemy = turns + additional_turns
        total_turns += turns_for_enemy
    print(total_turns)

if __name__ == '__main__':
    solve()