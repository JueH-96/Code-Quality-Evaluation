def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    total = sum(A)
    base = total // n
    r = total % n
    
    A.sort()
    
    ans = 0
    for i in range(n - r):
        if A[i] < base:
            ans += base - A[i]
            
    for i in range(n - r, n):
        if A[i] < base + 1:
            ans += (base + 1) - A[i]
            
    print(ans)

if __name__ == "__main__":
    main()