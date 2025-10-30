def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    X = int(data[index])
    index += 1
    Y = int(data[index])
    index += 1
    
    P = []
    T = []
    for _ in range(N - 1):
        P.append(int(data[index]))
        index += 1
        T.append(int(data[index]))
        index += 1
    
    Q = int(data[index])
    index += 1
    queries = [int(data[index + i]) for i in range(Q)]
    
    # Precompute the cumulative sum of T for faster access if needed
    # But in this approach, we calculate step-by-step for each query
    
    for q in queries:
        current_time = q + X
        for i in range(N - 1):
            # Wait for the next bus
            if current_time % P[i] == 0:
                depart_time = current_time
            else:
                depart_time = ((current_time // P[i]) + 1) * P[i]
            # Travel to next stop
            current_time = depart_time + T[i]
        # Walk from stop N to Aoki's house
        current_time += Y
        print(current_time)

if __name__ == "__main__":
    main()