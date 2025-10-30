# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    # The number of edges to remove to make it a forest
    # is the number of excess edges over N-1
    excess_edges = M - (N - 1)
    
    # If excess_edges is negative or zero, it means we have a forest or less than needed edges
    result = max(0, excess_edges)
    
    print(result)