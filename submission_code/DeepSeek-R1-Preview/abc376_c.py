def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:]))
    
    A_sorted = sorted(A)
    B_sorted = sorted(B)
    
    # Compute prefix array
    prefix = [False] * (N + 1)
    prefix[0] = True
    for i in range(1, N + 1):
        if i - 1 < len(B_sorted):
            if prefix[i - 1] and (A_sorted[i - 1] <= B_sorted[i - 1]):
                prefix[i] = True
        else:
            prefix[i] = False
    
    # Compute suffix array
    suffix = [False] * N
    if N >= 1:
        suffix[N - 1] = True
    for i in range(N - 2, -1, -1):
        if i < len(B_sorted):
            if (A_sorted[i + 1] <= B_sorted[i]) and suffix[i + 1]:
                suffix[i] = True
        else:
            suffix[i] = False  # This case should not occur
    
    # Collect candidates
    candidates = []
    for i in range(N):
        if prefix[i] and suffix[i]:
            candidates.append(A_sorted[i])
    
    if candidates:
        print(min(candidates))
    else:
        print(-1)

if __name__ == '__main__':
    main()