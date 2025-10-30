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
        person_index = task[0] - 1 # 0-indexed person
        target_pos = task[1]
        current_pos = positions[person_index]
        
        if current_pos == target_pos:
            continue
            
        if target_pos > current_pos:
            # Move east
            displacement = target_pos - current_pos
            d = [0] * n
            d[person_index] = displacement
            for j in range(person_index + 1, n):
                d[j] = max(0, positions[j-1] + d[j-1] + 1 - positions[j])
            task_moves = sum(d[person_index:])
            total_moves += task_moves
            for j in range(person_index, n):
                positions[j] += d[j]
        else:
            # Move west
            displacement = target_pos - current_pos
            d = [0] * n
            d[person_index] = displacement
            for j in range(person_index - 1, -1, -1):
                d[j] = min(0, positions[j+1] + d[j+1] - 1 - positions[j])
            task_moves = sum(abs(val) for val in d[:person_index+1])
            total_moves += task_moves
            for j in range(person_index + 1):
                positions[j] += d[j]
                
    print(total_moves)

if __name__ == '__main__':
    solve()