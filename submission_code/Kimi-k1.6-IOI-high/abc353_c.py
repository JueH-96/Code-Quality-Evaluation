import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    A.sort()
    sum_A = sum(A)
    mod = 10**8
    C = 0
    
    for i in range(N):
        target = mod - A[i]
        j = bisect.bisect_left(A, target, i + 1, N)
        C += (N - j)
    
    ans = sum_A * (N - 1) - mod * C
    print(ans)

if __name__ == "__main__":
    main()