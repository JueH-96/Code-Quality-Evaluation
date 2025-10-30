def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M, K = map(int, data[:3])
    
    index = 3
    tests = []
    
    # Parse input and build a bitmask for each test
    for _ in range(M):
        Ci = int(data[index])
        index += 1
        used_keys = list(map(int, data[index:index+Ci]))
        index += Ci
        R = data[index]
        index += 1
        
        bitmask = 0
        for k in used_keys:
            bitmask |= (1 << (k - 1))
            
        tests.append((bitmask, R))
    
    # Function to count set bits (popcount) in an integer
    def popcount(x):
        return bin(x).count('1')
    
    # Try all subsets of {1, 2, ..., N}
    # Subset represented by a bitmask in [0, 2^N - 1]
    ALL = 1 << N
    count = 0
    for subset in range(ALL):
        valid = True
        for bitmask, result in tests:
            # Count how many keys in subset are among used_keys for this test
            c = popcount(subset & bitmask)
            if result == 'o':
                if c < K:  # Should open but didn't
                    valid = False
                    break
            else:  # 'x'
                if c >= K:  # Should not open but did
                    valid = False
                    break
        if valid:
            count += 1
    
    print(count)

# Do not forget to call main
if __name__ == "__main__":
    main()