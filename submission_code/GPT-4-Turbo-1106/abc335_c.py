import sys

def main():
    # Read the number of parts and queries
    N, Q = map(int, sys.stdin.readline().split())
    
    # Initialize the position of the dragon parts
    parts = [[i, 0] for i in range(N)]
    
    # Process each query
    for _ in range(Q):
        query = sys.stdin.readline().split()
        
        if query[0] == '1':
            # Move the head in the specified direction
            direction = query[1]
            if direction == 'R':
                parts[0][0] += 1
            elif direction == 'L':
                parts[0][0] -= 1
            elif direction == 'U':
                parts[0][1] += 1
            elif direction == 'D':
                parts[0][1] -= 1
            
            # Move the other parts to follow the head
            for i in range(1, N):
                parts[i], parts[i-1] = parts[i-1], parts[i]
        
        elif query[0] == '2':
            # Output the position of the specified part
            p = int(query[1]) - 1
            print(parts[p][0], parts[p][1])

if __name__ == "__main__":
    main()