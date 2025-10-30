# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = []
    for i in range(1, len(data), 2):
        A = int(data[i])
        B = int(data[i+1])
        edges.append((A, B))
    
    # We will remove pairs from the largest index down to the smallest
    result = []
    for i in range(N//2, 0, -1):
        # Remove the pair (2*i, 2*i-1)
        result.append((2*i, 2*i-1))
    
    for x, y in result:
        print(x, y)

main()