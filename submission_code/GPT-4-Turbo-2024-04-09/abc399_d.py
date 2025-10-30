def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = list(map(int, data[index:index + 2 * N]))
        index += 2 * N
        
        # Find positions of each number
        positions = {}
        for i in range(2 * N):
            if A[i] not in positions:
                positions[A[i]] = []
            positions[A[i]].append(i)
        
        # Check for valid pairs
        valid_pairs = 0
        for a in range(1, N + 1):
            pos_a = positions[a]
            if abs(pos_a[0] - pos_a[1]) == 1:
                continue  # a's occurrences are adjacent, skip
            
            for b in range(a + 1, N + 1):
                pos_b = positions[b]
                if abs(pos_b[0] - pos_b[1]) == 1:
                    continue  # b's occurrences are adjacent, skip
                
                # Check if we can swap to make both adjacent
                # We need to check if any of the positions can be swapped to make both adjacent
                # Possible swaps: (pos_a[0], pos_b[0]), (pos_a[0], pos_b[1]), (pos_a[1], pos_b[0]), (pos_a[1], pos_b[1])
                # We need to check if these swaps can result in both a and b being adjacent pairs
                # This is true if and only if the positions are such that they can be swapped to form contiguous blocks
                # Sort positions to simplify checking
                a1, a2 = sorted(pos_a)
                b1, b2 = sorted(pos_b)
                
                # Check if a and b can be adjacent after some swaps
                # Possible configurations after swaps:
                # - a1 a2 b1 b2
                # - a1 b1 a2 b2
                # - a1 b1 b2 a2
                # - b1 a1 a2 b2
                # - b1 a1 b2 a2
                # - b1 b2 a1 a2
                # We need to check if any of these configurations can be achieved by a single swap
                # We only need to check if they can be adjacent by checking the gaps between them
                if (a1 + 1 == b1 or a2 + 1 == b1 or b2 + 1 == a1 or b2 + 1 == a2):
                    valid_pairs += 1
        
        results.append(valid_pairs)
    
    for result in results:
        print(result)