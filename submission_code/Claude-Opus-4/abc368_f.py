def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i and n // i != n:
                divisors.append(n // i)
    return [d for d in divisors if d != n]

def compute_grundy(n, memo):
    if n in memo:
        return memo[n]
    
    if n == 1:
        memo[n] = 0
        return 0
    
    divisors = get_divisors(n)
    reachable = set()
    
    for d in divisors:
        reachable.add(compute_grundy(d, memo))
    
    # Find MEX (minimum excludant)
    mex = 0
    while mex in reachable:
        mex += 1
    
    memo[n] = mex
    return mex

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute Grundy numbers for all values
memo = {}
xor_sum = 0

for a in A:
    xor_sum ^= compute_grundy(a, memo)

# Determine winner
if xor_sum == 0:
    print("Bruno")
else:
    print("Anna")