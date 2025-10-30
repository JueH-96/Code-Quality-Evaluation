def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    # We'll collect all answers and print at once (faster in Python).
    answers = []
    
    for _ in range(t):
        N = int(input_data[idx]); idx += 1
        K = int(input_data[idx]); idx += 1
        A = list(map(int, input_data[idx:idx+N]))
        idx += N
        B = list(map(int, input_data[idx:idx+N]))
        idx += N
        
        # Build where-values-appear map for A.
        # pos_map[v] will hold all indices i where A[i] = v (0-based).
        pos_map = [[] for _ in range(N+1)]
        for i, val in enumerate(A):
            pos_map[val].append(i)
        
        # Sort each list of positions (total sorting cost ~ O(N log N))
        for v in range(N+1):
            if pos_map[v]:
                pos_map[v].sort()
        
        # Build groups for B as well, so we know where each value of B is needed.
        # group_B[v] will be the sorted list of indices i where B[i] = v.
        group_B = [[] for _ in range(N+1)]
        for i, val in enumerate(B):
            group_B[val].append(i)
        
        # A helper to check if all indices in "need" can be covered by positions "have".
        def can_cover(need, have, K, n):
            p_idx = 0
            size_have = len(have)
            for i in need:
                left_bound = i - K
                if left_bound < 0:
                    left_bound = 0
                right_bound = i + K
                if right_bound >= n:
                    right_bound = n - 1
                
                # Advance p_idx so that have[p_idx] >= left_bound
                while p_idx < size_have and have[p_idx] < left_bound:
                    p_idx += 1
                # If we've exhausted "have", can't cover i
                if p_idx == size_have:
                    return False
                # If the current have[p_idx] is > right_bound, no index can cover i
                if have[p_idx] > right_bound:
                    return False
                # Otherwise i is covered; do NOT advance p_idx because
                # the same position can cover future i's as well.
            return True
        
        possible = True
        # Check each distinct value appearing in B
        for val in range(1, N+1):
            if group_B[val]:
                # If B needs 'val' but A does not have it at all
                if not pos_map[val]:
                    possible = False
                    break
                # Check coverage
                if not can_cover(group_B[val], pos_map[val], K, N):
                    possible = False
                    break
        
        answers.append("Yes" if possible else "No")
    
    print("
".join(answers))