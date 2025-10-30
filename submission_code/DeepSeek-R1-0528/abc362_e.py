mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 1:
        print("1")
        return
        
    res = [0] * (n+1)
    res[1] = n % mod
    res[2] = (n * (n-1) // 2) % mod
    
    dp_prev = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dp_prev[i][j] = 1

    for L in range(3, n+1):
        dict_i = [dict() for _ in range(n)]
        for i in range(n):
            for k in range(i):
                if dp_prev[k][i] != 0:
                    a_val = A[k]
                    if a_val in dict_i[i]:
                        dict_i[i][a_val] = (dict_i[i][a_val] + dp_prev[k][i]) % mod
                    else:
                        dict_i[i][a_val] = dp_prev[k][i] % mod
                        
        dp_curr = [[0] * n for _ in range(n)]
        total_L = 0
        for j in range(n):
            for i in range(j):
                key = 2 * A[i] - A[j]
                if key in dict_i[i]:
                    count = dict_i[i][key]
                    dp_curr[i][j] = (dp_curr[i][j] + count) % mod
                    total_L = (total_L + count) % mod
                    
        res[L] = total_L % mod
        dp_prev = dp_curr
        
        if total_L == 0:
            break
            
    output = []
    for k in range(1, n+1):
        output.append(str(res[k] % mod))
    print(" ".join(output))

if __name__ == '__main__':
    main()