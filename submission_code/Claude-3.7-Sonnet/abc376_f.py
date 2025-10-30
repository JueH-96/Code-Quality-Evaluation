def main():
    N, Q = map(int, input().split())
    instructions = []
    for _ in range(Q):
        H, T = input().split()
        T = int(T)
        instructions.append((H, T))
    
    result = min_operations(N, instructions)
    print(result)

def min_operations(N, instructions):
    left_hand = 1  # Initial position of left hand
    right_hand = 2  # Initial position of right hand
    total_operations = 0
    
    def min_distance(a, b):
        """Calculate the minimum distance between a and b on a ring with N nodes."""
        return min((b - a) % N, (a - b) % N)
    
    def next_clockwise(i):
        """Get the next position when moving clockwise from i."""
        return i % N + 1
    
    for H, T in instructions:
        if H == "L":
            if left_hand == T:
                continue  # Already at the target
            
            if right_hand == T:
                # Need to move the right hand away first
                right_hand = next_clockwise(T)
                total_operations += 1  # One operation to move the right hand away
            
            # Move the left hand to the target
            dist = min_distance(left_hand, T)
            total_operations += dist
            left_hand = T
        
        elif H == "R":
            if right_hand == T:
                continue  # Already at the target
            
            if left_hand == T:
                # Need to move the left hand away first
                left_hand = next_clockwise(T)
                total_operations += 1  # One operation to move the left hand away
            
            # Move the right hand to the target
            dist = min_distance(right_hand, T)
            total_operations += dist
            right_hand = T
    
    return total_operations

if __name__ == "__main__":
    main()