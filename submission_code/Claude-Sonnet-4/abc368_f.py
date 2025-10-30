def get_grundy(n, memo):
    if n in memo:
        return memo[n]
    
    if n == 1:
        memo[n] = 0
        return 0
    
    # Find all proper divisors (divisors except n itself)
    divisors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.add(i)  # i is a divisor
            if i != n // i:
                divisors.add(n // i)  # n//i is also a divisor
    
    # Remove n itself to get proper divisors
    divisors.discard(n)
    
    # Calculate mex of grundy numbers of all reachable states
    reachable_grundy = set()
    for d in divisors:
        reachable_grundy.add(get_grundy(d, memo))
    
    # Find mex (minimum excludant)
    mex = 0
    while mex in reachable_grundy:
        mex += 1
    
    memo[n] = mex
    return mex

n = int(input())
a = list(map(int, input().split()))

memo = {}
total_grundy = 0

for num in a:
    total_grundy ^= get_grundy(num, memo)

if total_grundy == 0:
    print("Bruno")
else:
    print("Anna")