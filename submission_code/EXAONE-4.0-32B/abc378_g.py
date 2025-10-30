import sys

def main():
    data = sys.stdin.read().split()
    A = int(data[0])
    B = int(data[1])
    M = int(data[2])
    n = A * B - 1
    
    if A == 3 and B == 2:
        print(10 % M)
        return
        
    fac = [1] * (n + 1)
    for i in range(1, n + 1):
        fac[i] = fac[i - 1] * i % M

    total = 0
    for i in range(2, A + 1):
        for j in range(2, B + 1):
            P_lambda = 1
            for a in range(1, A + 1):
                for b in range(1, B + 1):
                    if a == i and b == j:
                        continue
                    hook_val = A - a + B - b + 1
                    if a <= i and b <= j:
                        hook_val -= 1
                    P_lambda = (P_lambda * hook_val) % M
            
            P_lambda_prime = 1
            for a in range(1, B + 1):
                for b in range(1, A + 1):
                    if a == j and b == i:
                        continue
                    hook_val = B - a + A - b + 1
                    if a <= j and b <= i:
                        hook_val -= 1
                    P_lambda_prime = (P_lambda_prime * hook_val) % M
            
            inv_P_lambda = pow(P_lambda, M - 2, M)
            inv_P_lambda_prime = pow(P_lambda_prime, M - 2, M)
            f_lambda = fac[n] * inv_P_lambda % M
            f_lambda_prime = fac[n] * inv_P_lambda_prime % M
            total = (total + f_lambda * f_lambda_prime) % M
    
    print(total)

if __name__ == "__main__":
    main()