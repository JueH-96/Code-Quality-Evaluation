def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    P = [0] * (n + 1)
    for i in range(1, n + 1):
        P[i] = P[i - 1] ^ A[i - 1]
        
    adj_sum = sum(A)
    total_val = 0
    for b in range(32):
        cnt = 0
        for num in P:
            if (num >> b) & 1:
                cnt += 1
        total_val += cnt * (n + 1 - cnt) * (1 << b)
        
    ans = total_val - adj_sum
    print(ans)

if __name__ == '__main__':
    main()