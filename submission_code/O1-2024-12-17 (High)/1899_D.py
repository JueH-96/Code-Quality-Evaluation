def main():
    import sys
    from collections import defaultdict
    
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    pos = 1
    
    # We will process each test case in O(n) time
    # Observing that b_i^(b_j) = b_j^(b_i) for b_i = 2^a_i, b_j = 2^a_j
    # is equivalent to a_i*2^(a_j) = a_j*2^(a_i).
    # One can show that the only solutions are when a_i = a_j
    # or {a_i, a_j} = {1, 2}.
    
    # Hence we just need to count:
    # 1) Pairs of identical a-values.
    # 2) Pairs where one is 1 and the other is 2.
    
    # Implementation outline:
    # - We'll keep a frequency dictionary to count duplicates (for part 1).
    # - We'll also keep counters for how many 1s and 2s have appeared so far
    #   to quickly count pairs of type (1,2) and (2,1) while scanning.
    
    results = []
    for _ in range(t):
        n = int(data[pos]); pos += 1
        arr = list(map(int, data[pos:pos+n]))
        pos += n
        
        freq = defaultdict(int)
        same_pairs = 0  # to count a_i = a_j
        c1 = 0          # how many 1's seen so far
        c2 = 0          # how many 2's seen so far
        cross_12 = 0    # to count pairs (1,2) or (2,1)
        
        for x in arr:
            # Count how many previous elements are exactly x
            same_pairs += freq[x]
            freq[x] += 1
            
            # If x is 1 or 2, count cross pairs
            if x == 1:
                # Each previously seen 2 forms a (2,1) pair with the current
                cross_12 += c2
                c1 += 1
            elif x == 2:
                # Each previously seen 1 forms a (1,2) pair with the current
                cross_12 += c1
                c2 += 1
        
        results.append(str(same_pairs + cross_12))
    
    print("
".join(results))

# Do not forget to call main.
main()