def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 0:
        print(0)
        return
        
    L = [1] * n
    R = [1] * n
    
    for i in range(1, n):
        L[i] = min(A[i], L[i-1] + 1)
    
    for i in range(n-2, -1, -1):
        R[i] = min(A[i], R[i+1] + 1)
        
    ans = 0
    for i in range(n):
        k = min(L[i], R[i])
        if k > ans:
            ans = k
            
    print(ans)

if __name__ == "__main__":
    main()