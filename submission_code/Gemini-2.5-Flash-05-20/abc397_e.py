import sys
from collections import defaultdict

sys.setrecursionlimit(2 * 10**5 + 100) # Set recursion limit for deep trees

def solve():
    N, K = map(int, sys.stdin.readline().split())
    
    # Adjacency list for the tree
    adj = defaultdict(list)
    for _ in range(N * K - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # dfs(u, p) returns:
    #   -1: impossible to decompose (indicates a failure in current subtree)
    #    0: u and its subtree successfully decomposed, and u itself does not need to extend any path upwards
    #    >0 (X < K): u is part of a path segment of length X, and this segment needs to extend upwards through u
    #              (i.e., u is an endpoint of this segment, which needs more nodes to reach length K)
    
    def dfs(u, p):
        children_path_rem = [] # Stores lengths of path segments from children that need to extend upwards

        for v in adj[u]:
            if v == p:
                continue
            
            rem_v = dfs(v, u)
            if rem_v == -1:
                return -1 # Propagate impossibility from child's subtree
            if rem_v > 0:
                children_path_rem.append(rem_v)
        
        # Create a frequency map for path segments from children
        counts = defaultdict(int)
        for l in children_path_rem:
            counts[l] += 1
        
        segments_to_match_with_u = [] # Segments that u must attach to and extend upwards
        internal_node_matches_count = 0 # Number of K-length paths completed with u as an internal node

        # Iterate through possible segment lengths (from 1 up to K-1) to find matches.
        # We iterate over sorted keys to ensure consistent behavior, though not strictly required for correctness
        # if using frequency map carefully. The key is to avoid processing a complement value 'comp'
        # if it has already been processed as 'l'.
        
        keys_to_process = sorted(counts.keys())
        
        for l in keys_to_process:
            if counts[l] == 0: # If this length has been used up by pairing with its complement
                continue
            
            # Case 1: u acts as an internal node for a path of length K: (l -- u -- (K-1-l))
            # Total length: l + 1 (for u) + (K-1-l) = K.
            complement_for_internal = K - 1 - l 
            
            if complement_for_internal < 1: 
                # This segment 'l' is too long to form an internal path through 'u' 
                # where u connects to another child (K-1-l would be 0 or negative).
                # So, this segment must be one that 'u' carries upwards.
                segments_to_match_with_u.extend([l] * counts[l])
                counts[l] = 0 # Mark as processed
                continue

            if l == complement_for_internal: # e.g., K=3, l=1; then 1--u--1
                num_pairs = counts[l] // 2
                counts[l] %= 2 # Remaining odd segment, if any
                internal_node_matches_count += num_pairs
            else: # l != complement_for_internal
                if complement_for_internal in counts:
                    num_pairs = min(counts[l], counts[complement_for_internal])
                    counts[l] -= num_pairs
                    counts[complement_for_internal] -= num_pairs
                    internal_node_matches_count += num_pairs
            
            # Any remaining segments of length 'l' in 'counts' could not be paired up to form an internal path through 'u'.
            # These segments must be considered for extending upwards through 'u'.
            if counts[l] > 0:
                segments_to_match_with_u.extend([l] * counts[l])
                counts[l] = 0 # Mark as processed

        # At this point, `internal_node_matches_count` indicates how many K-length paths were formed where `u` was an internal node.
        # `segments_to_match_with_u` contains segments from children that `u` could potentially extend upwards.
        
        path_to_extend_up = -1 # Initialize to an impossible state
        
        if len(segments_to_match_with_u) > 1:
            # A node `u` can only connect to one path segment if it's an endpoint of an upward path,
            # or two path segments if it's an internal node.
            # If it has more than one segment left over that needs to extend upwards, it's impossible.
            return -1 
        
        # Decide the fate of 'u' and its path segment
        if len(segments_to_match_with_u) == 0 and internal_node_matches_count == 0:
            # All child segments were either perfectly paired (K-1-l style) or there were no child segments.
            # 'u' is now free to start a new segment of length 1, extending upwards.
            path_to_extend_up = 1
        elif len(segments_to_match_with_u) == 1 and internal_node_matches_count == 0:
            # 'u' extends the single remaining child segment.
            path_to_extend_up = segments_to_match_with_u[0] + 1
        elif len(segments_to_match_with_u) == 0 and internal_node_matches_count == 1:
            # 'u' acted as an internal node for exactly one K-length path which completed within its subtree.
            # This path is fully formed and does not extend upwards through 'u'. So, 'u' returns 0.
            path_to_extend_up = 0
        else:
            # Remaining impossible scenarios:
            # - `len(segments_to_match_with_u) == 1` AND `internal_node_matches_count == 1`:
            #   `u` cannot be part of an upward-extending path AND an internal node for another completed path.
            # - `internal_node_matches_count > 1`:
            #   `u` cannot be an internal node for multiple distinct K-length paths.
            return -1 
        
        # Check the final length of the path segment involving 'u'
        if path_to_extend_up == K:
            return 0 # A path of length K was formed, and it completes at u. It does not extend upwards.
        elif path_to_extend_up < K:
            return path_to_extend_up # The segment needs to extend further upwards.
        else: # path_to_extend_up > K (should not happen with correct logic, but for safety)
            return -1 # Path somehow became too long, indicating an error or impossibility.

    # Start DFS from node 1 (arbitrary root), with parent 0 (sentinel)
    final_result = dfs(1, 0)
    
    if final_result == 0:
        print("Yes")
    else:
        print("No")

solve()