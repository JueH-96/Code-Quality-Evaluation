# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    X = list(map(int, data[idx:idx+N]))
    idx += N
    Q = int(data[idx])
    idx += 1
    tasks = []
    for _ in range(Q):
        T = int(data[idx])
        G = int(data[idx+1])
        tasks.append((T-1, G))  # converting to 0-based index
        idx += 2
    
    # Initialize the positions
    positions = X.copy()
    
    total_moves = 0
    
    for task in tasks:
        T, G = task
        current_pos = positions[T]
        if current_pos == G:
            continue
        # Determine the direction
        if G > current_pos:
            direction = 1
        else:
            direction = -1
        # Move the person to the target
        # First, find the range of positions that need to be adjusted
        # The person at T will move to G, and others will shift accordingly
        # The number of moves is |G - current_pos|
        # But, we need to ensure that no two persons are at the same position
        # So, we need to shift all persons between T and the target position
        # Let's find the new positions for all persons
        # The new positions will be sorted, and the person at T will be at G
        # So, we need to find the new positions for all persons
        # The new positions will be:
        # For persons before T: positions[i] remains the same if positions[i] < G
        # For persons after T: positions[i] remains the same if positions[i] > G
        # But, we need to ensure that no two persons are at the same position
        # So, we need to shift all persons between T and the target position
        # The number of moves is the sum of the absolute differences between the new positions and the old positions
        # So, we need to find the new positions for all persons
        # The new positions will be:
        # For persons before T: positions[i] remains the same if positions[i] < G
        # For persons after T: positions[i] remains the same if positions[i] > G
        # For persons at T: positions[T] = G
        # For persons between T and the target position: positions[i] = positions[i-1] + 1 if direction is positive, or positions[i+1] - 1 if direction is negative
        # So, we need to find the new positions for all persons
        # Let's create a new list of positions
        new_positions = positions.copy()
        new_positions[T] = G
        # Now, we need to adjust the positions of the other persons
        # For persons before T: if their position is >= G, they need to be shifted to the left
        # For persons after T: if their position is <= G, they need to be shifted to the right
        # So, we need to find the new positions for all persons
        # Let's iterate through the list and adjust the positions
        # First, handle the persons before T
        for i in range(T-1, -1, -1):
            if new_positions[i] >= new_positions[i+1]:
                new_positions[i] = new_positions[i+1] - 1
        # Then, handle the persons after T
        for i in range(T+1, N):
            if new_positions[i] <= new_positions[i-1]:
                new_positions[i] = new_positions[i-1] + 1
        # Now, calculate the total moves
        for i in range(N):
            total_moves += abs(new_positions[i] - positions[i])
        # Update the positions
        positions = new_positions
    
    print(total_moves)

if __name__ == "__main__":
    main()