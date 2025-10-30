def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = data[1]
    S_list = data[2:2+N]
    
    # Precompute the prefix and suffix matches for each S
    prefix = []
    suffix = []
    for S in S_list:
        # Compute the prefix match: the maximum number of characters of T that can be matched by S as a prefix
        p = 0
        for c in S:
            if p < len(T) and c == T[p]:
                p += 1
        prefix.append(p)
        
        # Compute the suffix match: the maximum number of characters of T that can be matched by S as a suffix
        s = len(T) - 1
        for c in reversed(S):
            if s >= 0 and c == T[s]:
                s -= 1
        suffix.append(len(T) - 1 - s)
    
    # Count the number of pairs (i,j) where prefix[i] + suffix[j] >= len(T)
    from collections import defaultdict
    prefix_count = defaultdict(int)
    for p in prefix:
        prefix_count[p] += 1
    
    suffix_count = defaultdict(int)
    for s in suffix:
        suffix_count[s] += 1
    
    total = 0
    for p in prefix_count:
        for s in suffix_count:
            if p + s >= len(T):
                total += prefix_count[p] * suffix_count[s]
    
    print(total)

if __name__ == "__main__":
    main()