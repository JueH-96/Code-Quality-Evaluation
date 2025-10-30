def get_min_dist(a, b, n):
    # Get minimum distance between two positions on ring
    dist = abs(a - b)
    return min(dist, n - dist)

def can_reach(start, target, other_hand, n):
    # Check if we can reach target from start without crossing other_hand
    if start == target:
        return True
    
    # Try clockwise
    curr = start
    while True:
        curr = (curr % n) + 1
        if curr == target:
            return True
        if curr == other_hand:
            break
    
    # Try counter-clockwise
    curr = start
    while True:
        curr = ((curr - 2) % n) + 1
        if curr == target:
            return True
        if curr == other_hand:
            break
            
    return False

def get_moves(start, target, other_hand, n):
    # Get minimum moves needed to reach target from start without crossing other_hand
    if start == target:
        return 0
        
    # Try both clockwise and counterclockwise
    clockwise = float('inf')
    counter_clockwise = float('inf')
    
    # Clockwise
    curr = start
    moves = 0
    while True:
        curr = (curr % n) + 1
        moves += 1
        if curr == target:
            clockwise = moves
            break
        if curr == other_hand:
            break
    
    # Counter-clockwise
    curr = start
    moves = 0
    while True:
        curr = ((curr - 2) % n) + 1
        moves += 1
        if curr == target:
            counter_clockwise = moves
            break
        if curr == other_hand:
            break
            
    return min(clockwise, counter_clockwise)

def main():
    # Read input
    n, q = map(int, input().split())
    instructions = []
    for _ in range(q):
        h, t = input().split()
        instructions.append((h, int(t)))
    
    # Initial positions
    left = 1
    right = 2
    total_moves = 0
    
    # Process each instruction
    for hand, target in instructions:
        if hand == 'L':
            if can_reach(left, target, right, n):
                total_moves += get_moves(left, target, right, n)
                left = target
        else:  # hand == 'R'
            if can_reach(right, target, left, n):
                total_moves += get_moves(right, target, left, n)
                right = target
    
    print(total_moves)

if __name__ == "__main__":
    main()