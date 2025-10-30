import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    
    # Precompute the consecutive pairs array A
    if n >= 2:
        A = [0] * (n - 1)
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                A[i] = 1
    else:
        A = []
    
    # Compute prefix sums
    prefix = [0] * n
    for i in range(1, n):
        if i - 1 < len(A):
            prefix[i] = prefix[i - 1] + A[i - 1]
        else:
            prefix[i] = prefix[i - 1]  # This case should not occur for n >=2
    
    # Process each query
    for _ in range(q):
        l, r = map(int, sys.stdin.readline().split())
        if l >= r:
            print(0)
        else:
            print(prefix[r - 1] - prefix[l - 1])

if __name__ == "__main__":
    main()