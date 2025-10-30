import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        
        A = list(map(int, data[index:index + N]))
        index += N
        B = list(map(int, data[index:index + N]))
        index += N
        
        # Sort indices based on A values
        indices = list(range(N))
        indices.sort(key=lambda x: A[x])
        
        # Minimize the expression
        min_value = float('inf')
        current_sum = 0
        min_heap = []
        
        for idx in indices:
            # Add the current B value to the heap
            heapq.heappush(min_heap, B[idx])
            current_sum += B[idx]
            
            # If the heap size exceeds K, remove the smallest element
            if len(min_heap) > K:
                current_sum -= heapq.heappop(min_heap)
            
            # If we have exactly K elements, calculate the potential minimum
            if len(min_heap) == K:
                max_A = A[idx]
                min_value = min(min_value, max_A * current_sum)
        
        results.append(min_value)
    
    # Output all results
    sys.stdout.write('
'.join(map(str, results)) + '
')