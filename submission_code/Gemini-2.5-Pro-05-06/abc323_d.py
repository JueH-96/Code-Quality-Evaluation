import collections
import heapq
import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Using defaultdict(int) means if a size is accessed but not present, its count is 0.
    slime_counts = collections.defaultdict(int)
    
    # Input S_i are distinct.
    for _ in range(N):
        s, c = map(int, sys.stdin.readline().split())
        slime_counts[s] = c
        
    pq = []
    # Initialize priority queue with sizes that have at least 2 slimes,
    # as these are initially eligible for synthesis.
    for s_val, c_val in slime_counts.items():
        if c_val >= 2:
            heapq.heappush(pq, s_val)
            
    while pq:
        s = heapq.heappop(pq)
        
        # Current count of slimes of size s.
        # This count is guaranteed to be >= 2 when s is popped, due to the logic:
        # 1. s was added to pq only if its count was >= 2.
        # 2. If s/2 was processed (s/2 < s, so processed earlier), it would add to slime_counts[s].
        #    The count slime_counts[s] would only increase or stay the same (if 0 slimes were added).
        #    It cannot decrease below 2 if it started at >= 2 due to operations on smaller sizes.
        c = slime_counts[s]
        
        # This check is theoretically redundant given the above, but harmless if kept.
        # If for some reason c became < 2 (e.g. if problem rules were different), this would be important.
        # if c < 2:
        #     continue
            
        num_new_slime_pairs = c // 2 # Number of pairs of slimes of size s that can be synthesized
        remaining_s_slimes = c % 2   # Slimes of size s left over after synthesis
        
        slime_counts[s] = remaining_s_slimes # Update count for size s
        
        target_s = s * 2 # New slimes will be of size 2*s
        
        # Before updating slime_counts[target_s], get its current count.
        # This is crucial for the logic of deciding whether to add target_s to PQ.
        old_count_at_target = slime_counts[target_s] 
        
        slime_counts[target_s] += num_new_slime_pairs # Add newly synthesized slimes
        
        # Add target_s to PQ only if its count transitions from <2 to >=2.
        # This ensures target_s is processed if it becomes eligible AND
        # avoids duplicate entries or re-processing if it was already eligible/in PQ.
        if slime_counts[target_s] >= 2 and old_count_at_target < 2:
            heapq.heappush(pq, target_s)
            
    ans = 0
    # Sum up all remaining slimes.
    # slime_counts.values() will include entries with 0 count if they were processed
    # and ended up with 0 slimes of that size. Summing 0s is fine.
    for c_val in slime_counts.values():
        ans += c_val
        
    print(ans)

if __name__ == '__main__':
    solve()