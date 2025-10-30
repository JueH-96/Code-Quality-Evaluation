# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    A = list(map(int, data[2:N+2]))
    queries = []
    for i in range(Q):
        idx = int(data[N+2 + 2*i]) - 1
        x = int(data[N+2 + 2*i + 1])
        queries.append((idx, x))
    
    # Set to track the current elements in A
    current_elements = set(A)
    
    # Find initial mex
    current_mex = 0
    while current_mex in current_elements:
        current_mex += 1
    
    results = []
    
    for idx, x in queries:
        old_value = A[idx]
        
        # Update the array
        A[idx] = x
        
        # Update the set
        if old_value in current_elements:
            current_elements.remove(old_value)
        current_elements.add(x)
        
        # Adjust current_mex
        if old_value == current_mex:
            while current_mex in current_elements:
                current_mex += 1
        elif x < current_mex:
            # No need to adjust current_mex if x is less than current_mex
            pass
        
        # Record the current_mex
        results.append(current_mex)
    
    # Print all results
    for result in results:
        print(result)