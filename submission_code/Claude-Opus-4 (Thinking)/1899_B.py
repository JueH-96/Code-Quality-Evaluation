def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return divisors

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    
    max_diff = 0
    
    # Get all divisors of n
    divisors = get_divisors(n)
    
    for k in divisors:
        num_trucks = n // k
        
        if num_trucks == 1:
            continue  # Only one truck, difference is 0
        
        # Calculate weight of each truck and track min/max
        max_weight = float('-inf')
        min_weight = float('inf')
        
        for i in range(num_trucks):
            weight = sum(a[i * k:(i + 1) * k])
            max_weight = max(max_weight, weight)
            min_weight = min(min_weight, weight)
        
        diff = max_weight - min_weight
        max_diff = max(max_diff, diff)
    
    print(max_diff)