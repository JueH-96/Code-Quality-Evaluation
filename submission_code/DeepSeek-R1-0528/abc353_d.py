mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    powers = [10**d for d in range(1, 11)]
    
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + A[i - 1]
    
    total = 0
    for j in range(n):
        d = len(str(A[j]))
        total += prefix[j] * powers[d - 1]
        total += j * A[j]
    
    total %= mod
    print(total)

if __name__ == "__main__":
    main()