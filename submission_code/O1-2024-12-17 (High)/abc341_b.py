def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    
    # Read conversion parameters
    idx = 1 + N
    S = [0]*(N-1)
    T = [0]*(N-1)
    for i in range(N-1):
        S[i] = int(data[idx])
        T[i] = int(data[idx+1])
        idx += 2
    
    # For each i, convert as many full lumps of S[i] as possible into T[i] of currency (i+1).
    for i in range(N-1):
        lumps = A[i] // S[i]
        A[i+1] += lumps * T[i]
        A[i] %= S[i]  # leftover currency i can't help with further conversions
    
    # The final amount of currency N is in A[N-1].
    print(A[N-1])

# Do not forget to call main().
if __name__ == "__main__":
    main()