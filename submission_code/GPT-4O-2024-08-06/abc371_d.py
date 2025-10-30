# YOUR CODE HERE
import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    # Number of villages
    N = int(data[index])
    index += 1
    
    # Coordinates of the villages
    X = list(map(int, data[index:index + N]))
    index += N
    
    # Number of villagers in each village
    P = list(map(int, data[index:index + N]))
    index += N
    
    # Number of queries
    Q = int(data[index])
    index += 1
    
    # Queries
    queries = []
    for _ in range(Q):
        L = int(data[index])
        R = int(data[index + 1])
        queries.append((L, R))
        index += 2
    
    # Process each query
    results = []
    for L, R in queries:
        # Find the range of indices in X that are within [L, R]
        left_index = bisect.bisect_left(X, L)
        right_index = bisect.bisect_right(X, R) - 1
        
        # Calculate the sum of villagers in this range
        if left_index <= right_index:
            total_villagers = sum(P[left_index:right_index + 1])
        else:
            total_villagers = 0
        
        results.append(total_villagers)
    
    # Print results for each query
    for result in results:
        print(result)