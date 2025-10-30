def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    
    # Function to get the sum of a segment
    def get_sum(i, j):
        print(f"? {i} {j}")
        sys.stdout.flush()
        T = int(sys.stdin.readline().strip())
        if T == -1:
            sys.exit()
        return T
    
    # Decompose the range [L, R] into segments of the form [2^i * j, 2^i * (j+1) - 1]
    # We need to find all such segments that are completely within [L, R] and cover the entire range
    # We can do this by finding the largest possible i for each segment
    
    # Initialize the sum
    total = 0
    
    # Start from the smallest segment and move upwards
    current = L
    while current <= R:
        # Find the largest i such that 2^i divides current and 2^i * (j+1) <= 2^N
        # and the segment [2^i * j, 2^i * (j+1) - 1] is within [L, R]
        i = 0
        while True:
            next_i = i + 1
            segment_size = 2 ** next_i
            j = current // segment_size
            l = segment_size * j
            r = l + segment_size - 1
            if r > R or l < L or r >= 2 ** N:
                break
            i = next_i
        # Now, i is the largest possible for the current position
        segment_size = 2 ** i
        j = current // segment_size
        l = segment_size * j
        r = l + segment_size - 1
        # Ensure the segment is within [L, R]
        if l < L or r > R:
            # If not, reduce i
            i -= 1
            segment_size = 2 ** i
            j = current // segment_size
            l = segment_size * j
            r = l + segment_size - 1
        # Get the sum of this segment
        sum_segment = get_sum(i, j)
        total = (total + sum_segment) % 100
        # Move to the next position
        current = r + 1
    
    # Output the result
    print(f"! {total}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()