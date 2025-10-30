# YOUR CODE HERE
import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    X = list(map(int, data[idx:idx+N]))
    idx += N
    P = list(map(int, data[idx:idx+N]))
    idx += N
    Q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        L = int(data[idx])
        R = int(data[idx+1])
        queries.append((L, R))
        idx += 2
    
    # Create a list of tuples (X_i, P_i) and sort by X_i
    villages = list(zip(X, P))
    villages.sort()
    
    # Extract sorted X and P
    sorted_X = [v[0] for v in villages]
    sorted_P = [v[1] for v in villages]
    
    # Compute prefix sums
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + sorted_P[i]
    
    # Process each query
    for L, R in queries:
        # Find the leftmost index >= L
        left = bisect.bisect_left(sorted_X, L)
        # Find the rightmost index <= R
        right = bisect.bisect_right(sorted_X, R)
        # Calculate the sum from prefix[left] to prefix[right]
        total = prefix[right] - prefix[left]
        print(total)

if __name__ == "__main__":
    main()