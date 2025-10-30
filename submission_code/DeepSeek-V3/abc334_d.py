# YOUR CODE HERE
import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    R = list(map(int, data[idx:idx+N]))
    idx += N
    queries = []
    for _ in range(Q):
        queries.append(int(data[idx]))
        idx += 1
    
    # Sort the R array
    R.sort()
    
    # Compute the prefix sum
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + R[i]
    
    # Process each query
    for x in queries:
        # Find the rightmost index where prefix[i] <= x
        pos = bisect.bisect_right(prefix, x) - 1
        print(pos)

if __name__ == "__main__":
    main()