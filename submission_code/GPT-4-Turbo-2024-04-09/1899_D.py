def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    from collections import defaultdict
    from math import gcd
    
    for _ in range(t):
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index+n]))
        index += n
        
        # We need to count pairs (i, j) such that b_i^{b_j} = b_j^{b_i}
        # This translates to (2^a_i)^(2^a_j) = (2^a_j)^(2^a_i)
        # Which simplifies to a_i * 2^a_j = a_j * 2^a_i
        # This is true if and only if a_i = a_j (since 2^x is strictly increasing and injective)
        
        # Count frequencies of each a_i
        freq = defaultdict(int)
        for num in a:
            freq[num] += 1
        
        # Calculate the number of valid pairs
        count = 0
        for value in freq.values():
            if value > 1:
                # Choose 2 out of value
                count += value * (value - 1) // 2
        
        results.append(str(count))
    
    print("
".join(results))