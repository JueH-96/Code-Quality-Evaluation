# YOUR CODE HERE
def main():
    n, q = map(int, input().split())
    
    # Initialize positions of the dragon parts
    positions = [(i, 0) for i in range(1, n+1)]
    
    for _ in range(q):
        query = input().split()
        
        if query[0] == '1':
            direction = query[1]
            head_x, head_y = positions[0]
            
            # Determine the new head position
            if direction == 'R':
                new_head_pos = (head_x + 1, head_y)
            elif direction == 'L':
                new_head_pos = (head_x - 1, head_y)
            elif direction == 'U':
                new_head_pos = (head_x, head_y + 1)
            elif direction == 'D':
                new_head_pos = (head_x, head_y - 1)
            
            # Each part moves to the position of the part in front of it
            old_pos = positions[0]
            positions[0] = new_head_pos
            
            for i in range(1, n):
                temp = positions[i]
                positions[i] = old_pos
                old_pos = temp
                
        elif query[0] == '2':
            p = int(query[1])
            x, y = positions[p-1]
            print(f"{x} {y}")

if __name__ == "__main__":
    main()