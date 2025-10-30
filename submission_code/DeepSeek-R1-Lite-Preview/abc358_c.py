import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    stands = data[2:2+N]
    
    # Convert each stand's string to a bitmask
    bitmasks = []
    for s in stands:
        mask = 0
        for j in range(M):
            if s[j] == 'o':
                mask |= (1 << j)
        bitmasks.append(mask)
    
    universe = (1 << M) - 1  # All flavors
    
    # Iterate through subset sizes from 1 to N
    for size in range(1, N+1):
        for subset in itertools.combinations(range(N), size):
            total = 0
            for stand in subset:
                total |= bitmasks[stand]
                if total == universe:
                    break
            if total == universe:
                print(size)
                return

if __name__ == "__main__":
    main()