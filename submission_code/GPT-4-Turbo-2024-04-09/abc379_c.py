def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))
    A = list(map(int, data[2+M:2+2*M]))
    
    # Dictionary to hold the initial stones in each cell
    stones = {}
    for i in range(M):
        stones[X[i]] = A[i]
    
    # Total stones needed and available
    total_stones_needed = N
    total_stones_available = sum(A)
    
    if total_stones_needed != total_stones_available:
        print(-1)
        return
    
    # Calculate the minimum operations required
    operations = 0
    excess_stones = 0
    
    for i in range(1, N+1):
        current_stones = stones.get(i, 0)
        if current_stones + excess_stones < 1:
            # Not enough stones to make this cell have one stone
            print(-1)
            return
        else:
            # Calculate how many stones are excess after filling this cell with one stone
            excess_stones += current_stones - 1
            if excess_stones < 0:
                # If excess is negative, we need to move stones to make it zero
                operations -= excess_stones  # Subtracting a negative number
                excess_stones = 0
    
    print(operations)

if __name__ == "__main__":
    main()