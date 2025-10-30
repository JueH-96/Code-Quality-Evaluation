def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        # Read n and the array a from input
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index+n]))
        index += n
        
        # Frequency count for each exponent noted in the array.
        freq = {}
        for x in a:
            freq[x] = freq.get(x, 0) + 1
        
        # Any two notes with the same exponent "a" always yield:
        # (2^(a))^(2^(a)) = (2^(a))^(2^(a))
        # So we count all pairs with equal exponents: choose 2 for each frequency.
        count_pairs = 0
        for cnt in freq.values():
            count_pairs += cnt * (cnt - 1) // 2
        
        # For pairs with distinct exponents, we have:
        # b_i = 2^(a_i) and b_j = 2^(a_j)
        # b_i^(b_j) = 2^(a_i * 2^(a_j)) and b_j^(b_i) = 2^(a_j * 2^(a_i))
        # They are equal if and only if a_i * 2^(a_j) = a_j * 2^(a_i).
        #
        # Let a_i = x and a_j = y with x < y.
        # Then the equality becomes:
        #    x * 2^(y - x) = y
        # Rewrite: x * (2^(y-x) - 1) = y - x.
        # It turns out that the only solution in positive integers is x = 1 and y = 2.
        #
        # Therefore, we add extra pairs for all occurrences of 1 and 2.
        cnt1 = freq.get(1, 0)
        cnt2 = freq.get(2, 0)
        count_pairs += cnt1 * cnt2
        
        results.append(str(count_pairs))
    
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()