def get_divisors(n):
    """
    Get all divisors of n.
    """
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    return divisors

def solve(A, N, K):
    max_val = max(A)
    count = [0] * (max_val + 1)
    
    # Compute the count of elements divisible by each number
    for a in A:
        for d in get_divisors(a):
            count[d] += 1
    
    results = []
    for a in A:
        # Find the largest divisor d such that count[d] >= K
        divisors = get_divisors(a)
        divisors.sort(reverse=True)
        for d in divisors:
            if count[d] >= K:
                results.append(d)
                break
    
    return results

# Read inputs
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Solve and print results
results = solve(A, N, K)
for result in results:
    print(result)