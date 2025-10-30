import sys

def main():
    MOD = 998244353
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if N == 0:
        return
    
    max_A = max(A)
    # Precompute divisors for all numbers up to max_A
    divisors = [[] for _ in range(max_A + 1)]
    for x in range(1, max_A + 1):
        for multiple in range(x, max_A + 1, x):
            divisors[multiple].append(x)
    
    sum_d = [0] * (max_A + 1)
    S = [0] * (N + 1)  # S[0] unused, S[1] is for j=1
    
    for j in range(1, N + 1):
        x = A[j - 1]
        ds = sorted(divisors[x], reverse=True)
        s_j = 0
        processed = []
        for d in ds:
            total = sum_d[d]
            # Subtract sum of sum_d[m] where m is a multiple of d and m > d and m is in ds
            for m in ds:
                if m > d and m % d == 0:
                    total -= sum_d[m]
                    if m in processed:
                        total += sum_d[m]  # This should not happen as m is processed earlier
            s_j += d * total
            s_j %= MOD
            processed.append(d)
        S[j] = s_j % MOD
        
        # Update sum_d for all d in ds
        power = pow(2, j - 1, MOD)
        for d in ds:
            sum_d[d] = (sum_d[d] + power) % MOD
    
    # Compute the answers using the recurrence
    current_sum = 0
    ans = [0] * (N + 1)
    for m in range(2, N + 1):
        current_sum = (2 * current_sum + S[m]) % MOD
        ans[m] = current_sum
    
    for m in range(1, N + 1):
        print(ans[m] % MOD)

if __name__ == '__main__':
    main()