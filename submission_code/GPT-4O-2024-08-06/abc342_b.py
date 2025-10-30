# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = int(data[N+1])
    
    queries = []
    index = N + 2
    for _ in range(Q):
        A = int(data[index])
        B = int(data[index + 1])
        queries.append((A, B))
        index += 2
    
    # Create a dictionary to map person number to their position
    position_map = {P[i]: i for i in range(N)}
    
    results = []
    for A, B in queries:
        # Compare positions of A and B
        if position_map[A] < position_map[B]:
            results.append(A)
        else:
            results.append(B)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()