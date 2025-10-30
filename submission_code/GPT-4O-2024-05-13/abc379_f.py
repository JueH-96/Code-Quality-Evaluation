# YOUR CODE HERE
import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    heights = list(map(int, data[2:N+2]))
    queries = []
    for i in range(Q):
        l = int(data[N+2 + 2*i]) - 1
        r = int(data[N+2 + 2*i + 1]) - 1
        queries.append((l, r))
    
    results = []
    
    for l, r in queries:
        count = 0
        max_height = heights[r]
        for j in range(r + 1, N):
            if heights[j] > max_height:
                count += 1
                max_height = heights[j]
        results.append(count)
    
    for result in results:
        print(result)