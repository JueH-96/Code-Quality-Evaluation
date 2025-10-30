def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index + n]))
        index += n
        
        a.sort()
        
        max_diff = 0
        total_sum = sum(a)
        
        # Check all divisors of n
        for k in range(1, n + 1):
            if n % k == 0:
                # Calculate the sum of the first k elements (smallest)
                sum_min_k = sum(a[:k])
                # Calculate the sum of the last k elements (largest)
                sum_max_k = sum(a[-k:])
                
                # Calculate the absolute difference
                diff = abs(sum_max_k - sum_min_k)
                
                # Update the maximum difference
                max_diff = max(max_diff, diff)
        
        results.append(max_diff)
    
    for result in results:
        print(result)