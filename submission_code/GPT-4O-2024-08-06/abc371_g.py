# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    A = list(map(int, data[N+1:2*N+1]))
    
    visited = [False] * N
    result = [0] * N
    
    for i in range(N):
        if not visited[i]:
            # Find the cycle starting from i
            cycle_indices = []
            x = i
            while not visited[x]:
                visited[x] = True
                cycle_indices.append(x)
                x = P[x] - 1  # P is 1-based, convert to 0-based
            
            # Extract the cycle elements from A
            cycle_elements = [A[idx] for idx in cycle_indices]
            # Sort the cycle elements
            cycle_elements.sort()
            # Place them back in the result array
            for j, idx in enumerate(cycle_indices):
                result[idx] = cycle_elements[j]
    
    print(" ".join(map(str, result)))

main()