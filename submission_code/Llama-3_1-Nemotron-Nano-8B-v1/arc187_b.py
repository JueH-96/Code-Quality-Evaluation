MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))
    q = B.count(-1)
    
    current_product = [1] * (M + 2)  # 1-based indexing for x from 1 to M
    sum_ans = 0
    
    for i in range(N):
        if B[i] != -1:
            x = B[i]
            sum_ans = (sum_ans + current_product[x]) % MOD
        else:
            s = sum(current_product[1:M+1]) % MOD
            inv_M = pow(M, MOD-2, MOD)
            sum_ans = (sum_ans + s * inv_M) % MOD
        
        next_product = [0] * (M + 2)
        if B[i] == -1:
            inv_M = pow(M, MOD-2, MOD)
            for x in range(1, M+1):
                numerator = (M - x) % MOD
                p = numerator * inv_M % MOD
                next_product[x] = current_product[x] * p % MOD
        else:
            a = B[i]
            for x in range(1, M+1):
                p = 1 if x < a else 0
                next_product[x] = current_product[x] * p % MOD
        current_product = next_product
    
    pow_mq = pow(M, q, MOD)
    sum_total = sum_ans * pow_mq % MOD
    print(sum_total)

if __name__ == "__main__":
    main()