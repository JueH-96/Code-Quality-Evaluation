def solve():
    n, m, h, k = map(int, input().split())
    s = input()
    item_locations = set()
    for _ in range(m):
        x_i, y_i = map(int, input().split())
        item_locations.add((x_i, y_i))
    
    current_x, current_y = 0, 0
    current_health = h
    
    for move_direction in s:
        if move_direction == 'R':
            current_x += 1
        elif move_direction == 'L':
            current_x -= 1
        elif move_direction == 'U':
            current_y += 1
        elif move_direction == 'D':
            current_y -= 1
            
        current_health -= 1
        
        if current_health < 0:
            print("No")
            return
            
        current_position = (current_x, current_y)
        if current_position in item_locations:
            if current_health < k:
                current_health = k
                item_locations.remove(current_position)
                
    print("Yes")

if __name__ == '__main__':
    solve()