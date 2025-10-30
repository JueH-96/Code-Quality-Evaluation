def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    # Initialize the result array
    result = [0] * N
    
    # Use a stack to keep track of the indices of the buildings
    stack = []
    
    # Iterate from the end to the start
    for i in range(N-1, -1, -1):
        # Remove all buildings from the stack that are shorter than the current building
        while stack and H[i] > H[stack[-1]]:
            stack.pop()
        # The number of buildings that can be seen is the number of buildings in the stack
        result[i] = len(stack)
        # Push the current building's index onto the stack
        stack.append(i)
    
    # Print the result
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()