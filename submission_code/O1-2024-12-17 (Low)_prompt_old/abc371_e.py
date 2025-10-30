def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Positions array where pos[v] will hold the (1-based) indices where value v appears.
    # Since A_i ≤ N, we make an array of length N+1 to store positions.
    pos = [[] for _ in range(N+1)]
    for i, val in enumerate(A, start=1):
        pos[val].append(i)
    
    # Total number of subarrays in an array of length N
    total_sub = N * (N + 1) // 2

    answer = 0

    # For each possible value v, compute how many subarrays contain v at least once
    for v in range(1, N+1):
        if not pos[v]:
            continue
        
        # Calculate the total subarrays that do NOT contain v, by summing up the
        # subarrays fully contained in the "gaps" between occurrences of v.
        sum_gap_sub = 0
        
        # First gap: before the first occurrence
        first_occ = pos[v][0]
        gap_len = first_occ - 1
        sum_gap_sub += gap_len * (gap_len + 1) // 2
        
        # Gaps between consecutive occurrences
        for i in range(len(pos[v]) - 1):
            gap_len = pos[v][i+1] - pos[v][i] - 1
            sum_gap_sub += gap_len * (gap_len + 1) // 2
        
        # Last gap: after the last occurrence
        last_occ = pos[v][-1]
        gap_len = N - last_occ
        sum_gap_sub += gap_len * (gap_len + 1) // 2
        
        # Subarrays that contain v = total_sub - subarrays that do NOT contain v
        contain_v = total_sub - sum_gap_sub
        answer += contain_v
    
    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()