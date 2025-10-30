def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 0:
        print(0)
        return
    if n == 1:
        print(1)
        return
        
    D = []
    for i in range(1, n):
        D.append(A[i] - A[i-1])
        
    total = n
    i = 0
    while i < len(D):
        j = i
        while j < len(D) and D[j] == D[i]:
            j += 1
        L = j - i
        total += L * (L + 1) // 2
        i = j
        
    print(total)

if __name__ == "__main__":
    main()