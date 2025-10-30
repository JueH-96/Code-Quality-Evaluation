# YOUR CODE HERE
import sys
import bisect

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    A = list(map(int, data[index:index + N]))
    index += N
    
    queries = []
    for _ in range(Q):
        R = int(data[index])
        index += 1
        X = int(data[index])
        index += 1
        queries.append((R, X))
    
    results = []
    
    for R, X in queries:
        # Consider the first R elements of A
        relevant = [a for a in A[:R] if a <= X]
        
        # Find LIS in the filtered list
        lis = []
        for value in relevant:
            pos = bisect.bisect_left(lis, value)
            if pos < len(lis):
                lis[pos] = value
            else:
                lis.append(value)
        
        results.append(len(lis))
    
    for result in results:
        print(result)