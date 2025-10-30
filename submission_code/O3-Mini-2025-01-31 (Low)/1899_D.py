def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    
    # Explanation:
    # Each note is: b = 2^(a)
    # For two notes with exponents x and y (i.e. b_i = 2^x and b_j = 2^y),
    # the combination values are:
    # b_i^(b_j) = (2^x)^(2^y) = 2^(x * 2^y) and b_j^(b_i) = (2^y)^(2^x) = 2^(y * 2^x).
    # So equality holds if x * 2^y = y * 2^x.
    # Clearly, if x == y then the equality holds trivially.
    # By testing small exponents we see:
    #   For x = 1: 1*2^y, for y = 2: 1*4 = 4, and
    #   For x = 2: 2*2^1 = 2*2 = 4.
    # So (1, 2) and (2, 1) are also valid pairs.
    # In summary, if a_i are the exponents, the pairs (i, j) with a_i = a_j 
    # always work, plus any pair where one equals 1 and the other equals 2.
    
    # Process each test case
    for _ in range(t):
        n = int(data[index])
        index += 1
        
        # Read the exponents
        arr = list(map(int, data[index:index+n]))
        index += n
        
        # Count frequency of each exponent
        freq = {}
        for num in arr:
            freq[num] = freq.get(num, 0) + 1
        
        # Count valid pairs from same value: For each value, choose 2 among freq
        valid_pairs = 0
        for count in freq.values():
            if count > 1:
                valid_pairs += count * (count - 1) // 2
        
        # Add cross pairs from 1 and 2, if exist.
        if 1 in freq and 2 in freq:
            valid_pairs += freq[1] * freq[2]
            
        results.append(str(valid_pairs))
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()