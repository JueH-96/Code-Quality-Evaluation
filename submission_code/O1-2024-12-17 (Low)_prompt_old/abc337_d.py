def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, K = map(int, input_data[:3])
    grid_strings = input_data[3:]
    
    # Convert grid to a 2D list for easier handling
    grid = [list(row) for row in grid_strings]
    
    INF = 10**15
    min_operations = INF
    
    # Check rows for a horizontal run of length K
    for i in range(H):
        row = grid[i]
        # We'll scan segments separated by 'x'
        start = 0
        while start < W:
            if row[start] == 'x':
                start += 1
                continue
            # find a contiguous segment with no 'x'
            end = start
            while end < W and row[end] != 'x':
                end += 1
            # segment [start, end-1] has no 'x'
            length = end - start
            if length >= K:
                # build prefix sum of '.' inside this segment
                # prefix_sum[t] = number of '.' from segment start to (start + t - 1)
                prefix_sum = [0]*(length+1)
                for idx in range(length):
                    prefix_sum[idx+1] = prefix_sum[idx] + (1 if row[start+idx] == '.' else 0)
                # now check each sub-run of length K
                # sub-run from offset offset to offset+K-1 in this segment
                for offset in range(length - K + 1):
                    cost = prefix_sum[offset+K] - prefix_sum[offset]
                    if cost < min_operations:
                        min_operations = cost
            start = end
    
    # Check columns for a vertical run of length K
    for j in range(W):
        # We'll scan segments separated by 'x'
        start = 0
        while start < H:
            if grid[start][j] == 'x':
                start += 1
                continue
            # find contiguous segment with no 'x'
            end = start
            while end < H and grid[end][j] != 'x':
                end += 1
            # segment [start, end-1] has no 'x'
            length = end - start
            if length >= K:
                # build prefix sum of '.' in this column segment
                prefix_sum = [0]*(length+1)
                for idx in range(length):
                    prefix_sum[idx+1] = prefix_sum[idx] + (1 if grid[start+idx][j] == '.' else 0)
                # now check each sub-run of length K
                for offset in range(length - K + 1):
                    cost = prefix_sum[offset+K] - prefix_sum[offset]
                    if cost < min_operations:
                        min_operations = cost
            start = end
    
    # If we never found a valid run, min_operations will be INF
    if min_operations == INF:
        print(-1)
    else:
        print(min_operations)

def main():
    solve()

# Let's call solve()
if __name__ == "__main__":
    main()