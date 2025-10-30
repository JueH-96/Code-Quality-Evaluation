import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    total = [0] * (1 << n)
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += A[i]
        total[mask] = s
        
    dp = [set() for _ in range(1 << n)]
    dp[0].add(0)
    
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            rem = mask ^ sub
            for x in dp[rem]:
                candidate = total[sub] ^ x
                dp[mask].add(candidate)
            sub = (sub - 1) & mask
            
    print(len(dp[(1 << n) - 1]))

if __name__ == '__main__':
    main()