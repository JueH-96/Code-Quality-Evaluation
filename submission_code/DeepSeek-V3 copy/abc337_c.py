def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Create a dictionary to map each person to the person behind them
    behind = {}
    for i in range(N):
        if A[i] != -1:
            behind[A[i]] = i + 1  # since person numbers start from 1
    
    # Find the person at the front
    front = -1
    for i in range(N):
        if A[i] == -1:
            front = i + 1
            break
    
    # Reconstruct the line
    line = []
    current = front
    while current in behind:
        line.append(current)
        current = behind[current]
    line.append(current)
    
    print(' '.join(map(str, line)))

if __name__ == "__main__":
    main()