def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    # Convert each stand's flavor string to a bitmask
    s_bits = []
    for _ in range(N):
        s = input().strip()
        bit = 0
        for j, ch in enumerate(s):
            if ch == 'o':
                bit |= (1 << j)
        s_bits.append(bit)
    
    full = (1 << M) - 1  # bitmask where all M flavors are covered
    ans = N + 1
    
    # Try all subsets of stands
    for mask in range(1 << N):
        covered = 0
        for i in range(N):
            if mask & (1 << i):
                covered |= s_bits[i]
        # Check if this subset covers all flavors
        if covered == full:
            # Count how many stands are in this subset
            cnt = mask.bit_count()  # Python 3.8+
            ans = min(ans, cnt)
    
    print(ans)

if __name__ == "__main__":
    main()