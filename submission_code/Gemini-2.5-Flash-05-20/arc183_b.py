import sys
import collections
import bisect

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # possible[i] will be True if A[i] can be made equal to B[i]
    possible = [False] * N

    # q for BFS
    q = collections.deque()

    # val_to_B_indices[v] stores a sorted list of indices i where B[i] == v
    val_to_B_indices = collections.defaultdict(list)
    for i in range(N):
        val_to_B_indices[B[i]].append(i)

    # Initialize BFS queue with indices where A[i] == B[i]
    for i in range(N):
        if A[i] == B[i]:
            if not possible[i]: # Avoid adding duplicates if already processed
                possible[i] = True
                q.append(i)
    
    # Store the actual indices in a way that allows efficient removal
    # To avoid iterating processed indices repeatedly in the bisect range
    # We'll use a set for efficient checking and removal
    active_b_indices_for_val = collections.defaultdict(set)
    for val, indices_list in val_to_B_indices.items():
        for idx in indices_list:
            if not possible[idx]: # Only include if not already marked possible
                active_b_indices_for_val[val].add(idx)

    # Since we need sorted lists for bisect, and sets are unsorted, we need to adapt.
    # The `val_to_B_indices` list is already sorted, let's keep it that way.
    # We will just iterate over the relevant part of this list.
    # A cleaner approach would be to use a data structure that supports range query and deletion,
    # but `bisect` + a visited array for each list can still work.
    
    # To handle already processed indices and efficient removal for BFS:
    # Instead of removing from list (which is slow), just mark as visited.
    # If using lists for val_to_B_indices, need to be careful with iterating and modifying.
    # A set allows O(1) removal, but then iteration is not sorted.
    # Let's use `bisect` on `val_to_B_indices` (which are sorted lists) and check `possible` array.
    # To avoid re-iterating over already processed items in `val_to_B_indices[val]`,
    # we need to keep track of pointers for each value's list.
    
    # This optimization pattern is a bit complex for a standard BFS.
    # A simpler way to get O(N log N) is to have the lists, and when an index `j` is found
    # and marked `possible[j]=True`, remove it from its list `val_to_B_indices[B[j]]` 
    # to avoid re-finding it. But list removal is O(N).
    # We can create a temporary list of new candidates in each BFS step, then add them.

    while q:
        curr = q.popleft()
        val_curr = B[curr]

        # Get the sorted list of indices for val_curr
        indices_list = val_to_B_indices[val_curr]
        
        # Find the range of indices in indices_list that are within [curr - K, curr + K]
        # bisect_left finds insertion point for curr - K
        # bisect_right finds insertion point for curr + K + 1
        
        start_search_idx = bisect.bisect_left(indices_list, curr - K)
        end_search_idx = bisect.bisect_right(indices_list, curr + K)

        # Iterate through relevant indices in the list
        # Create a temporary list of indices to mark as possible and add to queue
        newly_possible_indices = []
        for i in range(start_search_idx, end_search_idx):
            neighbor_idx = indices_list[i]
            if not possible[neighbor_idx]:
                possible[neighbor_idx] = True
                newly_possible_indices.append(neighbor_idx)
        
        for idx in newly_possible_indices:
            q.append(idx)
            
    # Check if all positions could be made identical to B
    for i in range(N):
        if not possible[i]:
            sys.stdout.write("No
")
            return

    sys.stdout.write("Yes
")


T = int(sys.stdin.readline())
for _ in range(T):
    solve()