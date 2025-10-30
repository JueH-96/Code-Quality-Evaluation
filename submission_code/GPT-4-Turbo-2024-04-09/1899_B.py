def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    t = int(data[idx])
    idx += 1
    results = []
    
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        
        if n == 1:
            results.append(0)
            continue
        
        # Calculate the total weight of all boxes
        total_weight = sum(a)
        
        # Dictionary to store the sum of weights for each possible k
        weight_sums = {}
        
        # Calculate weight sums for each possible k
        for k in range(1, n + 1):
            if n % k == 0:  # Only consider k that perfectly divides n
                # Calculate the sum of weights for each group of k boxes
                group_sums = [sum(a[i:i+k]) for i in range(0, n, k)]
                max_diff = max(group_sums) - min(group_sums)
                weight_sums[k] = max_diff
        
        # Find the maximum difference among all valid k
        max_difference = max(weight_sums.values())
        results.append(max_difference)
    
    # Print all results for each test case
    for result in results:
        print(result)