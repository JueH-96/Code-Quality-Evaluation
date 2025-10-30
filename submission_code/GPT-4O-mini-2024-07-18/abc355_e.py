# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    
    total_sum = 0
    count = 0
    
    # We will ask questions in a way to cover the range [L, R]
    # We will use a binary approach to sum the segments
    for i in range(N + 1):  # i can go from 0 to N
        segment_size = 1 << i  # 2^i
        for j in range(0, (1 << N) // segment_size):
            l = j * segment_size
            r = (j + 1) * segment_size - 1
            
            # Check if the segment overlaps with [L, R]
            if r < L or l > R:
                continue
            
            # Adjust l and r to be within [L, R]
            l = max(l, L)
            r = min(r, R)
            
            # Ask the question
            print(f"? {i} {j}")
            sys.stdout.flush()
            response = int(input().strip())
            
            if response == -1:
                return  # Terminate if the response is -1
            
            total_sum += response
            count += 1
            
            # If we have enough information to determine the result, we can stop
            if count > 100:  # Arbitrary limit to prevent infinite loops
                break
    
    # Output the final result
    result = total_sum % 100
    print(f"! {result}")

if __name__ == "__main__":
    main()