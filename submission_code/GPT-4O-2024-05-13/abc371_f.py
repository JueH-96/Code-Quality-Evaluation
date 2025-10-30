# YOUR CODE HERE
def min_total_movements(N, X, Q, tasks):
    total_movements = 0
    
    for t, g in tasks:
        t -= 1  # Convert to 0-based index
        total_movements += abs(X[t] - g)
        X[t] = g  # Update the position of the t-th person
    
    return total_movements

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    N = int(data[index])
    index += 1
    
    X = list(map(int, data[index:index + N]))
    index += N
    
    Q = int(data[index])
    index += 1
    
    tasks = []
    for _ in range(Q):
        T = int(data[index])
        G = int(data[index + 1])
        tasks.append((T, G))
        index += 2
    
    result = min_total_movements(N, X, Q, tasks)
    print(result)