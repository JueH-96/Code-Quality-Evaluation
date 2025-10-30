# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    queries = list(map(int, data[2:2+Q]))
    
    # Possible number of fixed elements are all even numbers from 0 to N^2
    possible_fixed_elements = set(range(0, N*N + 1, 2))
    
    results = []
    for K in queries:
        if K in possible_fixed_elements:
            results.append("Yes")
        else:
            results.append("No")
    
    for result in results:
        print(result)