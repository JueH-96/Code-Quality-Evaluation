N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

# For each k, we need to find min((C*k + A[i]) % M) for all i
# This repeats with period M since both C and A[i] are < M
# So we can find the pattern in first M values and use that

# First find period length
period = 1
k = 1
seen = {0: [min((A[i]) % M for i in range(N))]}
curr_sum = seen[0][0]

while k < min(K, M):
    vals = [((C * k + A[i]) % M) for i in range(N)]
    min_val = min(vals)
    
    # Check if we've seen this pattern before
    pattern = tuple(vals)
    for prev_k, prev_vals in seen.items():
        if tuple(prev_vals) == pattern:
            period = k - prev_k
            break
    
    seen[k] = vals
    curr_sum += min_val
    
    if period != 1:
        break
    k += 1

if period == 1 or K <= k:
    # Either no period found or K is small enough
    ans = 0
    for kk in range(K):
        ans += min((C * kk + A[i]) % M for i in range(N))
else:
    # We found a period
    # Calculate complete periods
    complete_periods = (K - k) // period
    per_period_sum = sum(min((C * kk + A[i]) % M for i in range(N)) for kk in range(k-period, k))
    ans = curr_sum - sum(min((C * kk + A[i]) % M for i in range(N)) for kk in range(k-period, k))
    ans += complete_periods * per_period_sum
    
    # Calculate remaining
    remaining = K - k - complete_periods * period
    start = k + complete_periods * period
    for kk in range(start, K):
        ans += min((C * kk + A[i]) % M for i in range(N))

print(ans)