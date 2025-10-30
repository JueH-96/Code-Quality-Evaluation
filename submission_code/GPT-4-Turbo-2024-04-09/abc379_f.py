import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    heights = list(map(int, data[index:index+N]))
    index += N
    
    queries = []
    for _ in range(Q):
        l = int(data[index]) - 1
        index += 1
        r = int(data[index]) - 1
        index += 1
        queries.append((l, r))
    
    # To solve the problem efficiently, we need to preprocess the heights to determine
    # the visibility of buildings from any given building to the right.
    
    # We will use a stack to maintain the buildings that are visible as we scan from right to left.
    # This will help us answer the queries in constant time after preprocessing.
    
    # visible_from_right[i] will store the indices of buildings that are visible from building i to the right.
    visible_from_right = [[] for _ in range(N)]
    
    # Stack to keep track of buildings and their indices
    stack = []
    
    # Process each building from right to left
    for i in range(N-1, -1, -1):
        # Maintain the invariant that the stack is decreasing in height
        while stack and heights[stack[-1]] <= heights[i]:
            stack.pop()
        
        # All buildings in the stack are visible from building i
        visible_from_right[i] = stack[:]
        
        # Push current building onto the stack
        stack.append(i)
    
    # Now we need to answer each query
    results = []
    for l, r in queries:
        # We need to count buildings that are visible from both l and r and are to the right of r
        count = 0
        # Check buildings visible from r
        for visible_index in visible_from_right[r]:
            if visible_index > r and visible_index in visible_from_right[l]:
                count += 1
        results.append(count)
    
    # Output all results
    sys.stdout.write("
".join(map(str, results)) + "
")