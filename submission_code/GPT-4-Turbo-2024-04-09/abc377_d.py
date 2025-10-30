def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    intervals = []
    
    index = 2
    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        intervals.append((L, R))
        index += 2
    
    # We will use a difference array to mark the start and end of forbidden ranges
    diff = [0] * (M + 2)  # We use M+2 to safely handle boundaries
    
    for L, R in intervals:
        diff[L] += 1
        diff[R + 1] -= 1
    
    # Create the forbidden array using the prefix sum of the difference array
    forbidden = [0] * (M + 1)
    current = 0
    for i in range(1, M + 1):
        current += diff[i]
        forbidden[i] = current > 0
    
    # Calculate the number of valid (l, r) pairs
    valid_pairs = 0
    # We use a prefix sum to count how many r are valid for each l
    valid_r_count = [0] * (M + 1)
    
    for r in range(1, M + 1):
        valid_r_count[r] = valid_r_count[r - 1] + (0 if forbidden[r] else 1)
    
    for l in range(1, M + 1):
        if not forbidden[l]:
            # All r from l to M that are not forbidden are valid
            valid_pairs += valid_r_count[M] - valid_r_count[l - 1]
    
    print(valid_pairs)

if __name__ == "__main__":
    main()