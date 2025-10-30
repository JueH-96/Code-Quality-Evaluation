import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    
    queries = data[2:]
    
    # Initial positions of the dragon parts
    # Only the head position and the direction history is needed
    head_x = 1
    head_y = 0
    
    # To store the results of type 2 queries
    results = []
    
    # We need to track the movements of the head to calculate positions of other parts
    movements = []
    
    index = 0
    while index < len(queries):
        query_type = int(queries[index])
        if query_type == 1:
            # Move head
            direction = queries[index + 1]
            if direction == 'R':
                head_x += 1
            elif direction == 'L':
                head_x -= 1
            elif direction == 'U':
                head_y += 1
            elif direction == 'D':
                head_y -= 1
            movements.append(direction)
            index += 2
        elif query_type == 2:
            # Find coordinates of part p
            p = int(queries[index + 1])
            if p == 1:
                # Directly the head position
                results.append(f"{head_x} {head_y}")
            else:
                # Calculate the position of part p by retracing the movements
                # We need to go back p-1 steps in the movements list
                part_x = head_x
                part_y = head_y
                steps_back = p - 1
                move_count = len(movements)
                
                for i in range(steps_back):
                    if i >= move_count:
                        break
                    move = movements[move_count - 1 - i]
                    if move == 'R':
                        part_x -= 1
                    elif move == 'L':
                        part_x += 1
                    elif move == 'U':
                        part_y -= 1
                    elif move == 'D':
                        part_y += 1
                
                results.append(f"{part_x} {part_y}")
            index += 2
    
    # Output all results for type 2 queries
    print("
".join(results))

if __name__ == "__main__":
    main()