def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    sum_A = sum(A)
    m = sum_A // N
    rem = sum_A % N
    
    A.sort()
    res = 0
    
    for i in range(N):
        if i < (N - rem):
            target = m
        else:
            target = m + 1
        if A[i] > target:
            res += A[i] - target
    print(res)

if __name__ == "__main__":
    main()