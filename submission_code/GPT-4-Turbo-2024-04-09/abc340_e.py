def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Initialize the result array with the initial number of balls
    result = [0] * N
    
    # Process each operation
    for b in B:
        balls_in_hand = A[b]
        A[b] = 0  # Remove all balls from box B_i
        
        # Distribute the balls
        current_index = b
        while balls_in_hand > 0:
            current_index = (current_index + 1) % N
            result[current_index] += 1
            balls_in_hand -= 1
    
    # Print the result
    print(" ".join(map(str, result)))