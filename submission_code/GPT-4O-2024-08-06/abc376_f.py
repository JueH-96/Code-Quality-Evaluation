def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    instructions = []
    index = 2
    for _ in range(Q):
        H = data[index]
        T = int(data[index + 1])
        instructions.append((H, T))
        index += 2
    
    # Initial positions of the hands
    left_hand = 1
    right_hand = 2
    
    total_operations = 0
    
    for H, T in instructions:
        if H == 'L':
            # Move left hand to T
            if left_hand != T:
                # Calculate the minimum distance to move left hand to T
                clockwise_distance = (T - left_hand) % N
                counterclockwise_distance = (left_hand - T) % N
                move_left_hand = min(clockwise_distance, counterclockwise_distance)
                total_operations += move_left_hand
                left_hand = T
        elif H == 'R':
            # Move right hand to T
            if right_hand != T:
                # Calculate the minimum distance to move right hand to T
                clockwise_distance = (T - right_hand) % N
                counterclockwise_distance = (right_hand - T) % N
                move_right_hand = min(clockwise_distance, counterclockwise_distance)
                total_operations += move_right_hand
                right_hand = T
    
    print(total_operations)