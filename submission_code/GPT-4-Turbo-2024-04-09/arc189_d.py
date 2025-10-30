def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Result array
    B = [0] * N
    
    # We need to find the maximum size each slime can grow to.
    # We can use a stack to simulate the absorption process efficiently.
    
    # Process from left to right
    stack = []
    left_max = [0] * N  # This will store the maximum size achievable from the left side
    
    for i in range(N):
        current_size = A[i]
        
        # Absorb all smaller slimes from the stack
        while stack and A[stack[-1]] < current_size:
            idx = stack.pop()
            current_size += A[idx]
        
        left_max[i] = current_size
        stack.append(i)
    
    # Process from right to left
    stack = []
    right_max = [0] * N  # This will store the maximum size achievable from the right side
    
    for i in range(N-1, -1, -1):
        current_size = A[i]
        
        # Absorb all smaller slimes from the stack
        while stack and A[stack[-1]] < current_size:
            idx = stack.pop()
            current_size += A[idx]
        
        right_max[i] = current_size
        stack.append(i)
    
    # Combine results: the maximum size for each slime is the maximum of left and right absorptions
    for i in range(N):
        B[i] = max(left_max[i], right_max[i])
    
    # Print the result
    print(" ".join(map(str, B)))

if __name__ == "__main__":
    main()