# YOUR CODE HERE
def find_building_views():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    heights = list(map(int, data[1:]))
    
    # Result array to store the number of valid j's for each i
    result = [0] * N
    
    # Stack to keep track of the buildings that can "see" the current building
    stack = []
    
    # Traverse from right to left
    for i in range(N - 1, -1, -1):
        # Pop elements from the stack that are shorter than the current building
        while stack and heights[stack[-1]] < heights[i]:
            stack.pop()
        
        # The number of buildings that can be seen from building i is the number of buildings
        # in the stack, since they are all taller than the current building
        result[i] = len(stack)
        
        # Push the current building index onto the stack
        stack.append(i)
    
    # Print the result
    print(' '.join(map(str, result)))

find_building_views()