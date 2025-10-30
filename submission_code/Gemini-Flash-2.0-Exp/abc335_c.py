def solve():
    n, q = map(int, input().split())
    
    dragon = []
    for i in range(n):
        dragon.append([i + 1, 0])
    
    for _ in range(q):
        query = input().split()
        
        if query[0] == '1':
            direction = query[1]
            
            prev_x = dragon[0][0]
            prev_y = dragon[0][1]
            
            if direction == 'R':
                dragon[0][0] += 1
            elif direction == 'L':
                dragon[0][0] -= 1
            elif direction == 'U':
                dragon[0][1] += 1
            elif direction == 'D':
                dragon[0][1] -= 1
            
            for i in range(1, n):
                curr_x = dragon[i][0]
                curr_y = dragon[i][1]
                
                dragon[i][0] = prev_x
                dragon[i][1] = prev_y
                
                prev_x = curr_x
                prev_y = curr_y
        else:
            p = int(query[1])
            print(dragon[p-1][0], dragon[p-1][1])

solve()