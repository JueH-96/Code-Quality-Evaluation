def solve():
    n, q = map(int, input().split())
    
    positions = [(i, 0) for i in range(1, n + 1)]
    
    for _ in range(q):
        query = input().split()
        
        if query[0] == '1':
            direction = query[1]
            
            prev_head_pos = positions[0]
            
            if direction == 'R':
                positions[0] = (positions[0][0] + 1, positions[0][1])
            elif direction == 'L':
                positions[0] = (positions[0][0] - 1, positions[0][1])
            elif direction == 'U':
                positions[0] = (positions[0][0], positions[0][1] + 1)
            elif direction == 'D':
                positions[0] = (positions[0][0], positions[0][1] - 1)
            
            for i in range(1, n):
                temp = positions[i]
                positions[i] = prev_head_pos
                prev_head_pos = temp
                
        elif query[0] == '2':
            part = int(query[1])
            print(positions[part-1][0], positions[part-1][1])

solve()