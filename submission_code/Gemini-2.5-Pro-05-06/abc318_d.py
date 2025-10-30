import sys

def main():
    N = int(sys.stdin.readline())

    # adj[i][j] will store D_{i+1, j+1} for i < j (using 0-indexed i, j)
    adj = [[0] * N for _ in range(N)]
    for i in range(N - 1):  # For 0-indexed vertex i (from 0 to N-2)
        parts = list(map(int, sys.stdin.readline().split()))
        # parts contains weights for edges from vertex i to vertices i+1, i+2, ..., N-1
        for k in range(len(parts)):
            # The k-th element in parts is the weight of edge between
            # 0-indexed vertex i and 0-indexed vertex (i+1)+k
            u_0idx = i
            v_0idx = i + 1 + k 
            adj[u_0idx][v_0idx] = parts[k]

    # dp[mask] stores the maximum weight of a matching using a subset of vertices in 'mask'.
    # Not all vertices in 'mask' need to be matched.
    dp = [0] * (1 << N)  # dp[0] = 0 is the base case (empty matching, sum 0)

    for mask in range(1, 1 << N):
        # Pick 'p' as the smallest indexed vertex in 'mask' (LSB).
        p = (mask & -mask).bit_length() - 1
        
        # Option 1: vertex 'p' is not matched.
        # The max weight is then the same as for 'mask' without 'p'.
        # mask ^ (1 << p) is 'mask' with p-th bit flipped (removed, as it was set).
        dp[mask] = dp[mask ^ (1 << p)]

        # Option 2: vertex 'p' is matched with some other vertex 'j' in 'mask'.
        # Since 'p' is the LSB of 'mask', any other 'j' in 'mask' must have index j > p.
        # We iterate j from p+1 up to N-1.
        for j in range(p + 1, N):
            # Check if vertex 'j' is in 'mask'
            if (mask >> j) & 1:
                # If (p,j) form an edge:
                # Add its weight adj[p][j] (note p < j here).
                # Add the max matching weight from remaining vertices: mask ^ (1 << p) ^ (1 << j).
                current_val = adj[p][j] + dp[mask ^ (1 << p) ^ (1 << j)]
                
                if current_val > dp[mask]:
                    dp[mask] = current_val
                    
    # The final answer is for the mask containing all vertices.
    print(dp[(1 << N) - 1])

if __name__ == '__main__':
    main()