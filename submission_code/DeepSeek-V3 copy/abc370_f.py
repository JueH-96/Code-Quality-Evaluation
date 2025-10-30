def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Since the cake is circular, we can consider it as a linear array by duplicating it
    # to handle the circular nature.
    A_extended = A + A
    
    # Function to check if a given minimum weight is possible
    def is_possible(min_weight):
        count = 0
        current_sum = 0
        for a in A_extended:
            current_sum += a
            if current_sum >= min_weight:
                count += 1
                current_sum = 0
                if count >= K:
                    return True
        return False
    
    # Binary search to find the maximum possible minimum weight
    left = 0
    right = sum(A)
    max_min_weight = 0
    while left <= right:
        mid = (left + right) // 2
        if is_possible(mid):
            max_min_weight = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Now, find the number of cut lines that are never cut in any valid division
    # We need to find the number of cut lines that are always within a single person's segment
    # in all valid divisions.
    
    # To find the number of such cut lines, we need to find the number of cut lines that are
    # not part of any partition that achieves the max_min_weight.
    
    # We can iterate through the cake and count the number of cut lines that are not used in any
    # valid partition.
    
    # First, find all possible partitions that achieve the max_min_weight
    # Since it's circular, we need to consider all starting points
    
    # To count the number of cut lines that are never cut, we can mark the cut lines that are
    # used in at least one partition, and then count the ones that are never marked.
    
    # Initialize a list to mark whether a cut line is used in any partition
    used = [False] * N
    
    # Iterate through all possible starting points
    for start in range(N):
        current_sum = 0
        cuts = []
        for i in range(start, start + N):
            current_sum += A_extended[i]
            if current_sum >= max_min_weight:
                cuts.append(i)
                current_sum = 0
                if len(cuts) >= K:
                    break
        if len(cuts) >= K:
            # Mark the cut lines used in this partition
            for cut in cuts:
                used[cut % N] = True
    
    # The number of cut lines that are never cut is the total number of cut lines minus the number of used cut lines
    never_cut = N - sum(used)
    
    print(max_min_weight, never_cut)

if __name__ == "__main__":
    main()