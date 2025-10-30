def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    q = int(data[1])
    s = data[2]
    
    # Precompute a prefix sum array for consecutive matches.
    # For each i from 1 to n-1, if s[i] equals s[i-1] then add 1.
    prefix = [0] * n
    for i in range(1, n):
        prefix[i] = prefix[i - 1] + (1 if s[i] == s[i - 1] else 0)
        
    # Process queries.
    # For a query with indices l and r (1-indexed), the answer is:
    # prefix[r - 1] - prefix[l - 1]
    result = []
    pos = 3
    for _ in range(q):
        l = int(data[pos])
        r = int(data[pos + 1])
        pos += 2
        result.append(str(prefix[r - 1] - prefix[l - 1]))
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()