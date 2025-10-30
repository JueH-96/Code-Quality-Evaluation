import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    total = sum(A)
    k = total // N
    r = total % N
    sorted_A = sorted(A)
    res = 0
    
    # Process the last r elements (k+1 group)
    for i in range(N - r, N):
        diff = sorted_A[i] - (k + 1)
        if diff > 0:
            res += diff
    
    # Process the first N - r elements (k group)
    for i in range(N - r):
        diff = sorted_A[i] - k
        if diff > 0:
            res += diff
    
    print(res)

if __name__ == "__main__":
    main()