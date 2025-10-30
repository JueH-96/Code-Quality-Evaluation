def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    
    N = int(data[idx])
    idx += 1
    X = int(data[idx])
    idx += 1
    Y = int(data[idx])
    idx += 1
    
    P = []
    T = []
    for _ in range(N - 1):
        P.append(int(data[idx]))
        idx += 1
        T.append(int(data[idx]))
        idx += 1
    
    Q = int(data[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        queries.append(int(data[idx]))
        idx += 1
    
    results = []
    
    for q in queries:
        current_time = q + X
        for i in range(N - 1):
            # Wait for the next bus at bus stop i+1
            if current_time % P[i] == 0:
                next_bus_time = current_time
            else:
                next_bus_time = current_time + (P[i] - current_time % P[i])
            
            # Travel to the next bus stop
            current_time = next_bus_time + T[i]
        
        # Walk from the last bus stop to Aoki's house
        current_time += Y
        
        results.append(current_time)
    
    for result in results:
        print(result)