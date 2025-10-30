def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    
    n = int(next(it))
    q = int(next(it))
    s = next(it)
    
    # Precompute prefix sum for adjacent equal-letter pairs.
    # prefix[i] holds the count of indices j (1-indexed) with 1 <= j <= i such that s[j] equals s[j+1].
    # We use 0-indexed Python string but adjust accordingly.
    prefix = [0] * n
    for i in range(1, n):
        prefix[i] = prefix[i - 1] + (1 if s[i] == s[i - 1] else 0)
    
    results = []
    for _ in range(q):
        # Read the query bounds.
        l = int(next(it))
        r = int(next(it))
        # Compute the count of adjacent equal pairs in S[l:r]
        # Here, since prefix is built on 0-indexed positions and l, r are 1-indexed,
        # the count in the substring is: prefix[r-1] - prefix[l-1].
        count = prefix[r - 1] - prefix[l - 1]
        results.append(str(count))
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()