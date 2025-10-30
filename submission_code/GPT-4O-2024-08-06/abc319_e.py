# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    # Read N, X, Y
    N = int(data[index])
    index += 1
    X = int(data[index])
    index += 1
    Y = int(data[index])
    index += 1
    
    # Read P_i and T_i
    P = []
    T = []
    for _ in range(N - 1):
        P.append(int(data[index]))
        index += 1
        T.append(int(data[index]))
        index += 1
    
    # Read Q
    Q = int(data[index])
    index += 1
    
    # Read q_i
    queries = []
    for _ in range(Q):
        queries.append(int(data[index]))
        index += 1
    
    # Process each query
    results = []
    for q in queries:
        current_time = q + X  # Time when he reaches bus stop 1
        for i in range(N - 1):
            # Find the next bus departure time
            if current_time % P[i] == 0:
                next_departure = current_time
            else:
                next_departure = ((current_time // P[i]) + 1) * P[i]
            # Update the current time to when he reaches the next stop
            current_time = next_departure + T[i]
        
        # Add the time to walk from bus stop N to Aoki's house
        current_time += Y
        results.append(current_time)
    
    # Print results
    for result in results:
        print(result)