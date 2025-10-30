MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute lengths of each A_i
    len_A = [len(str(a)) for a in A]
    
    # Compute a_i mod MOD and pow10 for each element
    a = [x % MOD for x in A]
    pow10 = [pow(10, l, MOD) for l in len_A]
    
    # Compute Sum1
    sum1 = 0
    prefix_sum = 0
    for j in range(N):
        sum1 = (sum1 + prefix_sum * pow10[j]) % MOD
        prefix_sum = (prefix_sum + a[j]) % MOD
    
    # Compute Sum2
    sum2 = 0
    for j in range(N):
        sum2 = (sum2 + a[j] * j) % MOD
    
    total = (sum1 + sum2) % MOD
    print(total)

if __name__ == '__main__':
    main()