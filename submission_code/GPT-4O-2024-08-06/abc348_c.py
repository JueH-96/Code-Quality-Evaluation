# YOUR CODE HERE
def max_min_deliciousness():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = []
    C = []
    
    for i in range(N):
        A.append(int(data[2 * i + 1]))
        C.append(int(data[2 * i + 2]))
    
    # Dictionary to store the minimum deliciousness for each color
    color_min_deliciousness = {}
    
    for i in range(N):
        color = C[i]
        deliciousness = A[i]
        
        if color in color_min_deliciousness:
            color_min_deliciousness[color] = min(color_min_deliciousness[color], deliciousness)
        else:
            color_min_deliciousness[color] = deliciousness
    
    # Find the maximum of the minimum deliciousness values
    result = max(color_min_deliciousness.values())
    
    print(result)