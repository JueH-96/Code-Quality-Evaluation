def min_total_movements(N, positions, Q, T, G):
    total_movements = 0
    
    for q in range(Q):
        person = T[q] - 1  # Adjust for 0-indexed
        target_position = G[q]
        
        # Save the current positions for calculating movements
        original_positions = positions.copy()
        
        # Move person to the target position
        positions[person] = target_position
        
        # Check and adjust positions to the left of 'person'
        for i in range(person-1, -1, -1):
            if positions[i] >= positions[i+1]:
                positions[i] = positions[i+1] - 1
        
        # Check and adjust positions to the right of 'person'
        for i in range(person+1, N):
            if positions[i] <= positions[i-1]:
                positions[i] = positions[i-1] + 1
        
        # Calculate movements for this task
        task_movements = sum(abs(positions[i] - original_positions[i]) for i in range(N))
        total_movements += task_movements
    
    return total_movements

# Read input
N = int(input())
X = list(map(int, input().split()))
Q = int(input())
T, G = [], []
for _ in range(Q):
    t, g = map(int, input().split())
    T.append(t)
    G.append(g)

# Calculate and print the answer
result = min_total_movements(N, X, Q, T, G)
print(result)