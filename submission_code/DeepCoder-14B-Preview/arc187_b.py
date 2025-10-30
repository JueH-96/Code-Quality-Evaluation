MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    B = list(map(int, input[2:2+N]))
    
    q = sum(1 for x in B if x == -1)
    
    sum_C = 0
    for i in range(N-1):
        a = 0
        if B[i] == -1:
            a += 1
        if B[i+1] == -1:
            a += 1
        
        if a == 0:
            # Both fixed
            if B[i] > B[i+1]:
                cnt = 1
            else:
                cnt = 0
        elif a == 1:
            # One fixed, one variable
            if B[i] != -1:
                # B[i] is fixed, B[i+1] is variable
                x = B[i]
                cnt = x - 1
            else:
                # B[i] is variable, B[i+1] is fixed
                y = B[i+1]
                cnt = M - y
        else:
            # Both variables
            cnt = (M * (M - 1)) // 2
        
        exponent = q - a
        contribution = cnt * pow(M, exponent, MOD)
        contribution %= MOD
        sum_C = (sum_C + contribution) % MOD
    
    Mq = pow(M, q, MOD)
    total = (Mq + sum_C) % MOD
    print(total)

if __name__ == '__main__':
    main()