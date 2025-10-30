def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        arr = list(map(int, data[index:index+n]))
        index += n
        
        freq = {}
        for x in arr:
            freq[x] = freq.get(x, 0) + 1
        
        # All pairs (i, j) with a_i == a_j always satisfy b_i^(b_j) = b_j^(b_i)
        count_pairs = 0
        for count in freq.values():
            if count > 1:
                count_pairs += count * (count - 1) // 2
        
        # In addition, we have the special cross case:
        # The equation b_i^(b_j) = b_j^(b_i) reduces to a_i*2^(a_j) = a_j*2^(a_i).
        # For distinct a_i and a_j, one may check that the only solution is (a, b) = (1, 2) (or (2,1)).
        # Thus, we add f(1) * f(2) to the count.
        count_pairs += freq.get(1, 0) * freq.get(2, 0)
        
        results.append(str(count_pairs))
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()