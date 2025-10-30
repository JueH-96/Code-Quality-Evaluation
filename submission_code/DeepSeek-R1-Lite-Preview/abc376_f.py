def minimal_distance(a, b, N):
    return min(abs(a - b), N - abs(a - b))

def is_other_hand_on_path(start, target, other_hand, N, clockwise=True):
    if clockwise:
        if target >= start:
            return start < other_hand <= target
        else:
            return other_hand > start or other_hand <= target
    else:
        if target <= start:
            return start > other_hand >= target
        else:
            return other_hand < start or other_hand >= target

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    instructions = data[2:]
    
    L = 1
    R = 2
    total_operations = 0
    
    for i in range(Q):
        H = instructions[2*i]
        T = int(instructions[2*i+1])
        
        if H == 'L':
            specified_hand = L
            other_hand = R
        else:
            specified_hand = R
            other_hand = L
        
        distance_clockwise = (T - specified_hand) % N
        distance_counter = (specified_hand - T) % N
        minimal_dist = min(distance_clockwise, distance_counter)
        
        if distance_clockwise <= distance_counter:
            clockwise = True
            on_path = is_other_hand_on_path(specified_hand, T, other_hand, N, clockwise=True)
        else:
            clockwise = False
            on_path = is_other_hand_on_path(specified_hand, T, other_hand, N, clockwise=False)
        
        if on_path:
            total_operations += minimal_dist + 1
        else:
            total_operations += minimal_dist
        
        if H == 'L':
            L = T
        else:
            R = T
    
    print(total_operations)

if __name__ == "__main__":
    main()