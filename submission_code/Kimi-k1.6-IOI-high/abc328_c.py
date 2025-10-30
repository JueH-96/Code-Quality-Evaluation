def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    S = data[ptr]
    ptr += 1
    
    # Compute duplicates array
    duplicates = []
    for i in range(N-1):
        if S[i] == S[i+1]:
            duplicates.append(1)
        else:
            duplicates.append(0)
    
    # Compute prefix sums
    prefix = [0] * (N)  # prefix[0] = 0, prefix[1] = duplicates[0], etc.
    for i in range(1, N):
        if i-1 < len(duplicates):
            prefix[i] = prefix[i-1] + duplicates[i-1]
        else:
            prefix[i] = prefix[i-1]
    
    # Process queries
    for _ in range(Q):
        l = int(data[ptr])
        ptr +=1
        r = int(data[ptr])
        ptr +=1
        if l >= r:
            print(0)
        else:
            a = l -1
            b = r -2
            print(prefix[b+1] - prefix[a])
    
if __name__ == "__main__":
    main()