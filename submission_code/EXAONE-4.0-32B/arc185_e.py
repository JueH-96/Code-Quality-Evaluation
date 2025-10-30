MOD = 998244353
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    MAX_VAL = 100000
    N_max = 500000
    
    phi = list(range(MAX_VAL+1))
    for i in range(2, MAX_VAL+1):
        if phi[i] == i:
            for j in range(i, MAX_VAL+1, i):
                phi[j] -= phi[j] // i
                
    divisors = [[] for _ in range(MAX_VAL+1)]
    for i in range(1, MAX_VAL+1):
        for j in range(i, MAX_VAL+1, i):
            divisors[j].append(i)
            
    pow2 = [1] * (N_max+1)
    for i in range(1, N_max+1):
        pow2[i] = (pow2[i-1] * 2) % MOD
        
    G = [0] * (MAX_VAL+1)
    ans = [0] * n
    
    for i in range(n):
        x = A[i]
        total_add = 0
        for d in divisors[x]:
            if d <= MAX_VAL:
                total_add = (total_add + phi[d] * G[d]) % MOD
                
        if i == 0:
            current_ans = total_add
        else:
            current_ans = (2 * ans[i-1] + total_add) % MOD
        ans[i] = current_ans
        
        for d in divisors[x]:
            if d <= MAX_VAL:
                G[d] = (G[d] + pow2[i]) % MOD
                
    for res in ans:
        print(res)

if __name__ == "__main__":
    main()