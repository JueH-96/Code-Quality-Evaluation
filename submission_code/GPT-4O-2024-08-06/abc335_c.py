# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, Q = map(int, data[0].split())
    
    # Initialize the positions of the dragon parts
    positions = [(i, 0) for i in range(1, N + 1)]
    
    # Prepare to capture the output for type 2 queries
    output = []
    
    # Process each query
    for query in data[1:]:
        parts = query.split()
        if parts[0] == '1':
            # Move the head in the specified direction
            direction = parts[1]
            head_x, head_y = positions[0]
            
            if direction == 'R':
                new_head_position = (head_x + 1, head_y)
            elif direction == 'L':
                new_head_position = (head_x - 1, head_y)
            elif direction == 'U':
                new_head_position = (head_x, head_y + 1)
            elif direction == 'D':
                new_head_position = (head_x, head_y - 1)
            
            # Move each part to the position of the part in front of it
            for i in range(N - 1, 0, -1):
                positions[i] = positions[i - 1]
            
            # Update the head position
            positions[0] = new_head_position
        
        elif parts[0] == '2':
            # Output the position of part p
            p = int(parts[1]) - 1
            output.append(f"{positions[p][0]} {positions[p][1]}")
    
    # Print all the results for type 2 queries
    print("
".join(output))

if __name__ == "__main__":
    main()