import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + n]))
    ptr += n
    B = list(map(int, input[ptr:ptr + (n-1)]))
    
    A.sort()
    B.sort()
    
    # Compute prefix array
    prefix = [True] * (n + 1)
    for k in range(1, n):
        prefix[k] = prefix[k-1] and (A[k-1] <= B[k-1])
    
    # Compute suffix array
    suffix = [True] * n
    for k in range(n-2, -1, -1):
        suffix[k] = suffix[k+1] and (A[k+1] <= B[k])
    
    # Collect valid candidates
    candidates = []
    for k in range(n):
        if prefix[k] and suffix[k]:
            candidates.append(A[k])
    
    if candidates:
        print(min(candidates))
    else:
        print(-1)

if __name__ == "__main__":
    main()