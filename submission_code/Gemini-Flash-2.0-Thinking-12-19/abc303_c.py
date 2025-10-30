def solve():
    n, m, h, k = map(int, input().split())
    s = input()
    item_locations = set()
    for _ in range(m):
        x_i, y_i = map(int, input().split())
        item_locations.add((x_i, y_i))
    
    current_x, current_y = 0, 0
    current_health = h
    
    for move_char in s:
        if move_char == 'R':
            current_x += 1
        elif move_char == 'L':
            current_x -= 1
        elif move_char == 'U':
            current_y += 1
        elif move_char == 'D':
            current_y -= 1
            
        current_health -= 1
        
        if current_health < 0:
            print("No")
            return
            
        if (current_x, current_y) in item_locations and current_health < k:
            current_health = k
            
    print("Yes")

if __name__ == '__main__':
    solve()