def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    # Initialize the positions of the parts
    positions = [(i, 0) for i in range(1, N+1)]
    
    index = 2
    for _ in range(Q):
        query_type = data[index]
        if query_type == '1':
            C = data[index+1]
            # Move the head
            x, y = positions[0]
            if C == 'R':
                positions[0] = (x+1, y)
            elif C == 'L':
                positions[0] = (x-1, y)
            elif C == 'U':
                positions[0] = (x, y+1)
            elif C == 'D':
                positions[0] = (x, y-1)
            # Move the other parts
            for i in range(1, N):
                positions[i] = positions[i-1]
            index += 2
        else:
            p = int(data[index+1])
            x, y = positions[p-1]
            print(x, y)
            index += 2

if __name__ == "__main__":
    main()