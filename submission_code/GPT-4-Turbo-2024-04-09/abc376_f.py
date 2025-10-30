def distance_on_ring(N, start, end):
    """ Calculate the minimum distance on a ring of size N from start to end """
    if start <= end:
        direct = end - start
        wrap = start + N - end
    else:
        direct = N - start + end
        wrap = start - end
    return min(direct, wrap)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    queries = []
    index = 2
    for _ in range(Q):
        hand = data[index]
        target = int(data[index + 1])
        queries.append((hand, target))
        index += 2
    
    # Initial positions of hands
    left_pos = 1
    right_pos = 2
    
    total_operations = 0
    
    for hand, target in queries:
        if hand == 'L':
            # Move left hand to target
            if left_pos != target:
                # Calculate the minimum moves to get left hand to target
                if right_pos != target:
                    move_cost = distance_on_ring(N, left_pos, target)
                    left_pos = target
                else:
                    # If right hand is at the target, move right hand away first
                    # Move right hand clockwise or counterclockwise 1 step
                    if (right_pos % N) + 1 != left_pos:
                        new_right_pos = (right_pos % N) + 1
                    else:
                        new_right_pos = (right_pos - 2 + N) % N + 1
                    move_cost = distance_on_ring(N, left_pos, target) + 1
                    right_pos = new_right_pos
                    left_pos = target
                total_operations += move_cost
        elif hand == 'R':
            # Move right hand to target
            if right_pos != target:
                # Calculate the minimum moves to get right hand to target
                if left_pos != target:
                    move_cost = distance_on_ring(N, right_pos, target)
                    right_pos = target
                else:
                    # If left hand is at the target, move left hand away first
                    # Move left hand clockwise or counterclockwise 1 step
                    if (left_pos % N) + 1 != right_pos:
                        new_left_pos = (left_pos % N) + 1
                    else:
                        new_left_pos = (left_pos - 2 + N) % N + 1
                    move_cost = distance_on_ring(N, right_pos, target) + 1
                    left_pos = new_left_pos
                    right_pos = target
                total_operations += move_cost
    
    print(total_operations)