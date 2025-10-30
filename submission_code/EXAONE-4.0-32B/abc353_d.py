mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    L = [len(str(x)) for x in A]
    
    max_d = 10
    power = [0] * (max_d + 1)
    for d in range(1, max_d + 1):
        power[d] = pow(10, d, mod)
        
    total = 0
    curr_sum = 0
    for j in range(n):
        l_val = L[j]
        term1 = curr_sum * power[l_val] % mod
        term2 = A[j] * j % mod
        total = (total + term1 + term2) % mod
        curr_sum = (curr_sum + A[j]) % mod
        
    print(total)

if __name__ == '__main__':
    main()