def get_divisors_sum(n):
    divisors_sum = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors_sum += i
            if i * i != n:
                divisors_sum += n // i
        i += 1
    return divisors_sum

def is_good(n):
    return get_divisors_sum(n) % 3 == 0

MOD = 998244353

def solve(N, M):
    # Find all good numbers up to N
    good_nums = []
    for i in range(1, min(N + 1, 10**6)):
        if is_good(i):
            good_nums.append(i)
    
    # For larger N, check only numbers that could be factors
    if N >= 10**6:
        for i in range(10**6, int(N**0.5) + 1):
            if is_good(i):
                good_nums.append(i)
    
    # Count sequences whose product is good and ≤ N
    dp = {}
    def count_sequences(pos, prod):
        if pos == M:
            return 1 if prod in good_nums else 0
        if prod > N:
            return 0
        
        key = (pos, prod)
        if key in dp:
            return dp[key]
        
        result = 0
        # Try each possible factor
        i = 1
        while i * prod <= N:
            result = (result + count_sequences(pos + 1, i * prod)) % MOD
            i += 1
            
        dp[key] = result
        return result
    
    return count_sequences(0, 1)

N, M = map(int, input().split())
print(solve(N, M))