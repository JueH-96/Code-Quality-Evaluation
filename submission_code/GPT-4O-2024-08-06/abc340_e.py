# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    # Initialize the result array with the initial number of balls
    result = A[:]
    
    for b in B:
        # Take all balls from box B_i
        balls_in_hand = result[b]
        result[b] = 0
        
        # Distribute the balls
        for c in range(balls_in_hand):
            result[(b + c + 1) % N] += 1
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()