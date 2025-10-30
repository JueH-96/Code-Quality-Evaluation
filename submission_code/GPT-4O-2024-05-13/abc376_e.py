# YOUR CODE HERE
import heapq
import sys
input = sys.stdin.read

def solve():
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    
    for _ in range(T):
        N = int(data[idx])
        K = int(data[idx + 1])
        idx += 2
        A = list(map(int, data[idx:idx + N]))
        idx += N
        B = list(map(int, data[idx:idx + N]))
        idx += N
        
        # Pair A and B with their indices
        AB = [(A[i], B[i]) for i in range(N)]
        # Sort by A values
        AB.sort()
        
        # Minimize the expression
        min_value = float('inf')
        current_sum = 0
        max_heap = []
        
        for i in range(N - 1, -1, -1):
            a, b = AB[i]
            heapq.heappush(max_heap, -b)
            current_sum += b
            
            if len(max_heap) > K:
                current_sum += heapq.heappop(max_heap)
            
            if len(max_heap) == K:
                min_value = min(min_value, a * current_sum)
        
        results.append(min_value)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')