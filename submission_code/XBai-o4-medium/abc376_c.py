import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    A.sort()
    B.sort()
    
    prefix = [True] * N
    for k in range(1, N):
        prefix[k] = prefix[k-1] and (A[k-1] <= B[k-1])
    
    suffix = [True] * N
    for k in range(N-2, -1, -1):
        suffix[k] = suffix[k+1] and (A[k+1] <= B[k])
    
    candidates = []
    for k in range(N):
        if prefix[k] and suffix[k]:
            candidates.append(A[k])
    
    if not candidates:
        print(-1)
    else:
        print(min(candidates))

if __name__ == "__main__":
    main()