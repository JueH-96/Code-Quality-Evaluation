# YOUR CODE HERE
import sys
input = sys.stdin.readline

def solve():
    N = int(input())
    X = list(map(int, input().split()))
    Q = int(input())
    tasks = [tuple(map(int, input().split())) for _ in range(Q)]
    
    # Initialize the positions of the persons
    positions = X[:]
    
    # Initialize the total number of movements
    total_movements = 0
    
    # Process each task
    for T, G in tasks:
        T -= 1  # Convert to 0-based index
        # Calculate the number of movements required for the T-th person to reach G
        movements = abs(positions[T] - G)
        total_movements += movements
        
        # Update the positions of the persons
        for i in range(N):
            if positions[i] < G:
                positions[i] += movements
            elif positions[i] > G:
                positions[i] -= movements
        
        # Update the position of the T-th person
        positions[T] = G
    
    print(total_movements)

solve()