def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    # To keep track of the number of balls to be added to each box
    add_balls = [0] * N
    
    for b in B:
        # Number of balls taken from box B[i]
        balls_taken = A[b]
        A[b] = 0  # Empty the box
        
        # Distribute the balls
        for c in range(balls_taken):
            target_box = (b + c + 1) % N
            add_balls[target_box] += 1
    
    # Add the distributed balls to the original boxes
    for i in range(N):
        A[i] += add_balls[i]
    
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()