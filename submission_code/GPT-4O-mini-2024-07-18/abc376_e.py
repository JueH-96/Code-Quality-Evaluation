def min_expression_value(test_cases):
    results = []
    
    for N, K, A, B in test_cases:
        # Combine A and B into pairs and sort by A values
        combined = sorted(zip(A, B))
        
        # We will maintain a max heap for the sum of the largest K B values
        import heapq
        
        # This will store the largest K B values
        max_heap = []
        sum_B = 0
        
        # Iterate through the sorted A values
        for i in range(N):
            a_i, b_i = combined[i]
            
            # Add the current B value to the max heap
            heapq.heappush(max_heap, b_i)
            sum_B += b_i
            
            # If we have more than K elements in the heap, pop the smallest
            if len(max_heap) > K:
                sum_B -= heapq.heappop(max_heap)
            
            # If we have exactly K elements, calculate the expression
            if len(max_heap) == K:
                result = a_i * sum_B
                results.append(result)
    
        # The minimum result for this test case
        results.append(min(results[-N:]))
    
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    T = int(data[0])
    index = 1
    test_cases = []
    
    for _ in range(T):
        N, K = map(int, data[index].split())
        A = list(map(int, data[index + 1].split()))
        B = list(map(int, data[index + 2].split()))
        test_cases.append((N, K, A, B))
        index += 3
    
    results = min_expression_value(test_cases)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()