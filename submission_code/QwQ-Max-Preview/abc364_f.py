import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    groups = []
    for _ in range(Q):
        L = int(input[idx])
        idx += 1
        R = int(input[idx])
        idx += 1
        C = int(input[idx])
        idx += 1
        groups.append((L, R, C))
    
    # Step 1: Check if all nodes 1..N are covered
    # Merge all intervals and check if they form [1, N]
    groups_all = sorted([(L, R) for L, R, C in groups], key=lambda x: (x[0], -x[1]))
    merged = []
    for L, R in groups_all:
        if not merged:
            merged.append((L, R))
        else:
            last_L, last_R = merged[-1]
            if L > last_R + 1:
                merged.append((L, R))
            else:
                new_L = last_L
                new_R = max(last_R, R)
                merged[-1] = (new_L, new_R)
    # Check if merged intervals form [1, N]
    is_covered = False
    if len(merged) == 1 and merged[0][0] == 1 and merged[0][1] == N:
        is_covered = True
    else:
        is_covered = False
    
    if not is_covered:
        print(-1)
        return
    
    # Step 2: Process groups in increasing order of C
    groups_sorted = sorted(groups, key=lambda x: x[2])
    merged_intervals = []
    total_cost = 0
    
    for L, R, C in groups_sorted:
        # Find all intervals in merged_intervals that overlap with [L, R]
        # Using bisect to find potential overlapping intervals
        overlaps = []
        # Find the first interval where start <= R
        # Using bisect right on the start of intervals
        left = 0
        right = len(merged_intervals)
        while left < right:
            mid = (left + right) // 2
            m_L, m_R = merged_intervals[mid]
            if m_L <= R:
                left = mid + 1
            else:
                right = mid
        # After loop, left is the first index where m_L > R
        # So check all intervals before left
        i = 0
        overlapping = []
        while i < left:
            m_L, m_R = merged_intervals[i]
            if m_R >= L and m_L <= R:
                overlapping.append((m_L, m_R))
            i += 1
        
        # Calculate the number of overlapping intervals
        count = len(overlapping)
        total_cost += C * count
        
        # Merge these intervals into a single interval
        if not overlapping:
            # Add new interval [L, R]
            # Find the position to insert
            pos = bisect.bisect_left([interval[0] for interval in merged_intervals], L)
            merged_intervals.insert(pos, (L, R))
        else:
            # Find the new merged interval
            min_L = min(L, overlapping[0][0])
            max_R = max(R, overlapping[-1][1])
            # Remove all overlapping intervals from merged_intervals
            # Find the indices to remove
            first = merged_intervals.index(overlapping[0])
            last = merged_intervals.index(overlapping[-1])
            del merged_intervals[first:last+1]
            # Insert the new merged interval
            new_interval = (min_L, max_R)
            # Find position to insert
            pos = bisect.bisect_left([interval[0] for interval in merged_intervals], min_L)
            merged_intervals.insert(pos, new_interval)
    
    # Check if the final merged intervals is [1, N]
    if len(merged_intervals) == 1 and merged_intervals[0][0] == 1 and merged_intervals[0][1] == N:
        print(total_cost)
    else:
        print(-1)

if __name__ == '__main__':
    main()