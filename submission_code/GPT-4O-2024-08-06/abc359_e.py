# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    A = [0] * (N + 1)
    result = [0] * N
    operations = 0
    
    while True:
        operations += 1
        A[0] += 1
        
        for i in range(1, N + 1):
            if A[i-1] > A[i] and A[i-1] > H[i-1]:
                A[i-1] -= 1
                A[i] += 1
            
            if A[i] > 0 and result[i-1] == 0:
                result[i-1] = operations
        
        if all(result):
            break
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()