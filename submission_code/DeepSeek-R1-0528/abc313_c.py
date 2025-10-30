def main():
    n = int(input().strip())
    A = list(map(int, input().split()))
    
    total = sum(A)
    base = total // n
    r = total % n
    
    A.sort(reverse=True)
    
    operations = 0
    for i in range(r):
        if A[i] > base + 1:
            operations += A[i] - (base + 1)
    
    for i in range(r, n):
        if A[i] > base:
            operations += A[i] - base
            
    print(operations)

if __name__ == '__main__':
    main()