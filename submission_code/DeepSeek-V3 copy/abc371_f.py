def main():
    import sys
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
        T = int(data[idx]) - 1
        G = int(data[idx+1])
        tasks.append((T, G))
        idx += 2
    
    # Initialize the positions
    positions = X.copy()
    
    total_moves = 0
    
    for T, G in tasks:
        current_pos = positions[T]
        target_pos = G
        if current_pos == target_pos:
            continue
        # Determine the direction
        if target_pos > current_pos:
            direction = 1
        else:
            direction = -1
        # Move the person step by step
        # To minimize the total moves, we need to ensure that no two persons are at the same position
        # So, we need to move all persons between the current and target positions
        # First, find the range of positions that need to be adjusted
        if direction == 1:
            # Moving east
            # Find all persons to the right of current_pos and left of target_pos
            # They need to be moved to the right to make space
            # The order is from right to left
            # So, we iterate from the end to the start
            for i in range(N-1, -1, -1):
                if positions[i] < target_pos and positions[i] >= current_pos:
                    # Move this person to the right
                    # The new position is positions[i] + 1
                    # But we need to ensure that it doesn't overlap with the next person
                    # So, we move it to the next available position
                    # The next available position is positions[i] + 1
                    # But if the next person is already at positions[i] + 1, we need to move it further
                    # So, we need to find the next available position
                    # The next available position is the maximum of positions[i] + 1 and the next person's position + 1
                    # But since we are iterating from the end, the next person is already processed
                    # So, we can safely move it to positions[i] + 1
                    new_pos = positions[i] + 1
                    # Update the position
                    positions[i] = new_pos
                    # Increment the total moves
                    total_moves += 1
        else:
            # Moving west
            # Find all persons to the left of current_pos and right of target_pos
            # They need to be moved to the left to make space
            # The order is from left to right
            # So, we iterate from the start to the end
            for i in range(N):
                if positions[i] > target_pos and positions[i] <= current_pos:
                    # Move this person to the left
                    # The new position is positions[i] - 1
                    # But we need to ensure that it doesn't overlap with the previous person
                    # So, we move it to the previous available position
                    # The previous available position is positions[i] - 1
                    # But if the previous person is already at positions[i] - 1, we need to move it further
                    # So, we need to find the previous available position
                    # The previous available position is the minimum of positions[i] - 1 and the previous person's position - 1
                    # But since we are iterating from the start, the previous person is already processed
                    # So, we can safely move it to positions[i] - 1
                    new_pos = positions[i] - 1
                    # Update the position
                    positions[i] = new_pos
                    # Increment the total moves
                    total_moves += 1
        # Now, move the target person to the target position
        # The target person is at current_pos, and we need to move it to target_pos
        # The number of moves is |target_pos - current_pos|
        # But since we have already moved all the persons in between, the target position is now available
        # So, we can directly move the target person
        moves = abs(target_pos - positions[T])
        positions[T] = target_pos
        total_moves += moves
    
    print(total_moves)

if __name__ == "__main__":
    main()