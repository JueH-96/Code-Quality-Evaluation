def solve():
    n = int(input())
    x = list(map(int, input().split()))
    q = int(input())
    tasks = []
    for _ in range(q):
        tasks.append(list(map(int, input().split())))
    
    positions = list(x)
    total_moves = 0
    
    for task in tasks:
        person_index = task[0]
        target_pos = task[1]
        person_current_pos = positions[person_index-1]
        
        while person_current_pos != target_pos:
            if person_current_pos < target_pos:
                # move east
                next_pos = person_current_pos + 1
                if person_index < n and positions[person_index] <= next_pos:
                    person_to_move = person_index + 1
                    while person_to_move <= n and positions[person_to_move-1] <= next_pos:
                        positions[person_to_move-1] += 1
                        total_moves += 1
                        person_to_move += 1
                else:
                    positions[person_index-1] += 1
                    total_moves += 1
            else:
                # move west
                next_pos = person_current_pos - 1
                if person_index > 1 and positions[person_index-2] >= next_pos:
                    person_to_move = person_index - 1
                    while person_to_move >= 1 and positions[person_to_move-1] >= next_pos:
                        positions[person_to_move-1] -= 1
                        total_moves += 1
                        person_to_move -= 1
                else:
                    positions[person_index-1] -= 1
                    total_moves += 1
            person_current_pos = positions[person_index-1]
            
    print(total_moves)

if __name__ == '__main__':
    solve()