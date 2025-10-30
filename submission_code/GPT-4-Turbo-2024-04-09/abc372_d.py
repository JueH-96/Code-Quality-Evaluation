def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    heights = list(map(int, data[1:]))
    
    results = [0] * N
    
    # We will use a stack to keep track of indices of buildings
    # where each building is the tallest seen so far from the right
    stack = []
    
    # Traverse from right to left
    for i in range(N-1, -1, -1):
        # Maintain the stack such that it contains indices of buildings
        # that are taller than all buildings to their right up to the current building
        while stack and heights[stack[-1]] < heights[i]:
            stack.pop()
        
        # The size of the stack is the count of buildings that can be seen from building i
        results[i] = len(stack)
        
        # Push current index onto the stack
        stack.append(i)
    
    print(' '.join(map(str, results)))

if __name__ == "__main__":
    main()