# YOUR CODE HERE
def solve():
    N = int(input())
    X = list(map(int, input().split()))
    Q = int(input())
    
    positions = X.copy()
    total_movements = 0
    
    for _ in range(Q):
        T, G = map(int, input().split())
        T -= 1  # Convert to 0-based index
        
        # Find the current position of person T in the sorted list
        current_index = sorted(range(N), key=lambda i: positions[i]).index(T)
        
        # Calculate the number of movements needed
        movements = abs(G - positions[T])
        
        # Update the position of person T
        positions[T] = G
        
        # Calculate additional movements needed to maintain order
        if G > positions[T]:
            for i in range(current_index + 1, N):
                if positions[sorted(range(N), key=lambda i: positions[i])[i]] <= G:
                    movements += G - positions[sorted(range(N), key=lambda i: positions[i])[i]] + 1
                    positions[sorted(range(N), key=lambda i: positions[i])[i]] = G + 1
                else:
                    break
        else:
            for i in range(current_index - 1, -1, -1):
                if positions[sorted(range(N), key=lambda i: positions[i])[i]] >= G:
                    movements += positions[sorted(range(N), key=lambda i: positions[i])[i]] - G + 1
                    positions[sorted(range(N), key=lambda i: positions[i])[i]] = G - 1
                else:
                    break
        
        total_movements += movements
    
    print(total_movements)

solve()