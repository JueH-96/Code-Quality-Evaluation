def main():
    import sys

    data = sys.stdin.read().strip().split()
    L, R = map(int, data)
    
    intervals = []
    current = L
    
    # A small helper to get the largest power of 2 <= n
    def largest_power_of_2_leq(n):
        # For n>0, this returns 2^(floor(log2(n)))
        return 1 << (n.bit_length() - 1)
    
    while current < R:
        if current == 0:
            # Choose the largest power of 2 that doesn't exceed (R - current)
            length = largest_power_of_2_leq(R - current)
        else:
            # Start with the largest power of 2 dividing 'current'
            # i.e. x & -x for x=current
            length = current & (-current)
            # Ensure it doesn't exceed what remains (R - current)
            while length > (R - current):
                length >>= 1
        
        intervals.append((current, current + length))
        current += length
    
    # Output
    print(len(intervals))
    for l, r in intervals:
        print(l, r)

# Call main() at the end
main()