import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    H, W = map(int, data[0].split())
    n = 1 << W
    F = [0] * n
    
    for i in range(1, 1+H):
        s = data[i].strip()
        x = 0
        for j in range(W):
            if s[j] == '1':
                x |= (1 << j)
        F[x] += 1

    G = [0] * n
    for i in range(n):
        cnt = bin(i).count('1')
        G[i] = min(cnt, W - cnt)
        
    def fwht(a, inverse=False):
        n_val = len(a)
        m = 1
        while m < n_val:
            step = 2 * m
            for i in range(0, n_val, step):
                for j in range(i, i+m):
                    x = a[j]
                    y = a[j+m]
                    a[j] = x + y
                    a[j+m] = x - y
            m = step
        if inverse:
            for i in range(n_val):
                a[i] //= n_val
                
    F1 = F[:]
    G1 = G[:]
    fwht(F1)
    fwht(G1)
    
    H_arr = [F1[i] * G1[i] for i in range(n)]
    
    fwht(H_arr, inverse=True)
    
    ans = min(H_arr)
    print(ans)

if __name__ == "__main__":
    main()