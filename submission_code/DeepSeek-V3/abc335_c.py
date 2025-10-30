# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    
    # Initialize the positions of the dragon parts
    # part 1 is the head, part N is the tail
    # Initially, part i is at (i, 0)
    positions = [(i, 0) for i in range(1, N+1)]
    
    output = []
    for _ in range(Q):
        query_type = data[idx]
        if query_type == '1':
            C = data[idx+1]
            idx += 2
            # Move the head
            x, y = positions[0]
            if C == 'R':
                x += 1
            elif C == 'L':
                x -= 1
            elif C == 'U':
                y += 1
            elif C == 'D':
                y -= 1
            # Update the head position
            positions[0] = (x, y)
            # Move the rest of the parts
            for i in range(1, N):
                positions[i] = positions[i-1]
        else:
            p = int(data[idx+1])
            idx += 2
            # Get the position of part p
            x, y = positions[p-1]
            output.append(f"{x} {y}")
    
    # Print all outputs
    print("
".join(output))

if __name__ == "__main__":
    main()