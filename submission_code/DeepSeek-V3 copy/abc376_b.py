# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    instructions = []
    index = 2
    for _ in range(Q):
        H = data[index]
        T = int(data[index+1])
        instructions.append((H, T))
        index += 2
    
    # Initial positions
    left = 1
    right = 2
    
    total_ops = 0
    
    for H, T in instructions:
        if H == 'L':
            # Move left hand to T, right hand remains
            current = left
            target = T
            # Calculate the minimal steps
            # Since it's a ring, we can go clockwise or counter-clockwise
            # But we cannot move to the position of the right hand
            # So, we need to ensure that the path does not include the right hand's position
            # Since the problem guarantees that T != right when H is L, we can proceed
            # Calculate the distance
            # Option 1: clockwise
            dist1 = (target - current) % N
            # Option 2: counter-clockwise
            dist2 = (current - target) % N
            # Choose the minimal distance
            ops = min(dist1, dist2)
            total_ops += ops
            left = target
        else:
            # Move right hand to T, left hand remains
            current = right
            target = T
            # Calculate the minimal steps
            # Since it's a ring, we can go clockwise or counter-clockwise
            # But we cannot move to the position of the left hand
            # So, we need to ensure that the path does not include the left hand's position
            # Since the problem guarantees that T != left when H is R, we can proceed
            # Calculate the distance
            # Option 1: clockwise
            dist1 = (target - current) % N
            # Option 2: counter-clockwise
            dist2 = (current - target) % N
            # Choose the minimal distance
            ops = min(dist1, dist2)
            total_ops += ops
            right = target
    
    print(total_ops)

if __name__ == "__main__":
    main()