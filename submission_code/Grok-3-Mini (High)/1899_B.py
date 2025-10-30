import sys
import math

def get_divisors(n):
    divs = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

data = sys.stdin.read().split()
index = 0
t = int(data[index])
index += 1

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n
    
    # Compute prefix sums
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + a[i]
    
    # Get all divisors of n
    divs = get_divisors(n)
    
    # Initialize answer for this test case
    ans = 0
    
    # Iterate over all possible k (divisors of n)
    for k in divs:
        m = n // k  # Number of trucks
        max_sum_k = float('-inf')
        min_sum_k = float('inf')
        
        # Compute the sum for each truck and find max and min sum
        for i in range(1, m + 1):
            sum_val = prefix[i * k] - prefix[(i - 1) * k]
            if sum_val > max_sum_k:
                max_sum_k = sum_val
            if sum_val < min_sum_k:
                min_sum_k = sum_val
        
        # Compute the difference for this k
        diff_k = max_sum_k - min_sum_k
        
        # Update the answer with the maximum difference
        if diff_k > ans:
            ans = diff_k
    
    # Output the answer for this test case
    print(ans)