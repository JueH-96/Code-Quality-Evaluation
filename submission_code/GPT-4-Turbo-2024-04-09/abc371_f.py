def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    N = int(data[index])
    index += 1
    positions = list(map(int, data[index:index+N]))
    index += N
    
    Q = int(data[index])
    index += 1
    
    tasks = []
    for _ in range(Q):
        T = int(data[index]) - 1
        G = int(data[index+1])
        index += 2
        tasks.append((T, G))
    
    total_moves = 0
    current_positions = positions[:]
    
    for T, G in tasks:
        # Calculate the distance the T-th person needs to move
        distance = abs(current_positions[T] - G)
        total_moves += distance
        # Update the position of the T-th person
        current_positions[T] = G
    
    print(total_moves)

if __name__ == "__main__":
    main()