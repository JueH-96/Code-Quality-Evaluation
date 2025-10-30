class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        import sys
        sys.setrecursionlimit(10**7)

        n = len(nums)
        m = len(changeIndices)

        # Quick impossibility checks:
        # 1) If sum(nums) + n > m, we can never fit all required decrements + marks.
        total_decrements = sum(nums)
        if total_decrements + n > m:
            return -1

        # 2) If an index i has nums[i] == 0 but never appears in changeIndices, it's impossible to mark it.
        #    Similarly, if nums[i] > 0 but i never appears in changeIndices, we also can't mark it.
        #    Either way, if i never appears, it's impossible.
        appears = [False]*(n+1)  # 1-indexed
        for c in changeIndices:
            appears[c] = True
        for i in range(1, n+1):
            if not appears[i]:
                # If nums[i-1] == 0 or > 0, in both cases we cannot mark it
                return -1

        # Precompute adjacency for bipartite matching:
        # We'll match "index i" (1..n) on the left to "time s" (1..m) on the right
        # if changeIndices[s-1] == i.  (Because changeIndices is 0-based in Python.)
        # We only need to build it once for the maximum time = m, and later we will
        # restrict ourselves to s <= T in the feasibility check.
        adj = [[] for _ in range(n+1)]
        for s in range(1, m+1):
            i = changeIndices[s-1]
            adj[i].append(s)  # time s is a candidate to mark index i

        # Hopkroft-Karp (or similar) for bipartite matching:
        #   left side = indices 1..n
        #   right side = times 1..m
        # matchL[i] = the right-side node matched to left-node i
        # matchR[s] = the left-side node matched to right-node s
        # If an edge (i,s) is valid only if s <= T, we'll just check s <= T in BFS/DFS.

        from collections import deque

        def bfs(T):
            """ Build the distance array dist[] for left-side nodes
                using only edges to times <= T. """
            queue = deque()
            for i in range(1, n+1):
                if matchL[i] == 0:
                    dist[i] = 0
                    queue.append(i)
                else:
                    dist[i] = float('inf')
            dist[0] = float('inf')

            while queue:
                i = queue.popleft()
                if dist[i] < dist[0]:
                    for s in adj[i]:
                        if s > T:
                            break  # times are in ascending order, so we can stop
                        # matched partner on the left side:
                        partner = matchR[s]
                        if dist[partner] == float('inf'):
                            dist[partner] = dist[i] + 1
                            queue.append(partner)
            return dist[0] != float('inf')

        def dfs(i, T):
            """ Try to find an augmenting path from left node i using edges to s <= T. """
            if i != 0:
                for s in adj[i]:
                    if s > T:
                        break
                    partner = matchR[s]
                    if dist[partner] == dist[i] + 1 and dfs(partner, T):
                        matchR[s] = i
                        matchL[i] = s
                        return True
                dist[i] = float('inf')
                return False
            return True

        def can_feasibly_mark_all(T):
            """Check if we can mark all indices by time T (≤ m),
               and also place all decrements before each index's marking time.
               
               Steps:
                 1) We need T >= total_decrements + n (necessary condition).
                 2) We need each index i to have at least one s <= T in adj[i].
                 3) Find a maximum bipartite matching among i in 1..n and s in 1..T
                    using edges (i -> s) if changeIndices[s-1] = i.
                    If we can't match all n indices, fail.
                 4) Suppose we get a perfect matching i -> s_i. Then we sort i by s_i ascending.
                 5) We have T total slots: we use exactly n of them for marking.
                    That leaves T-n slots for decrements. We then check if we can allocate
                    nums[i] decrements for each i in the free slots strictly before s_i.
                    We do a simple greedy: for i in ascending order of s_i, pick any
                    nums[i] free slots among [1..s_i-1]. If not enough, fail. Otherwise succeed.
            """
            if T < total_decrements + n:
                return False

            # 2) If for some i, all s in adj[i] are > T, we cannot mark i
            for i in range(1, n+1):
                # Because adj[i] is sorted ascending
                if not adj[i] or adj[i][0] > T:
                    return False

            # 3) Hopkroft-Karp to see if we can match all n indices with distinct s <= T
            # Reset matching
            for i in range(1, n+1):
                matchL[i] = 0
            for s in range(1, T+1):
                matchR[s] = 0
            
            matching_size = 0
            # BFS/DFS until no more augmenting path
            while bfs(T):
                for i in range(1, n+1):
                    if matchL[i] == 0:
                        if dfs(i, T):
                            matching_size += 1
            
            if matching_size < n:
                return False

            # 4) For each i, we have matchL[i] = s_i. Collect them and sort by s_i ascending.
            indexing = []
            for i in range(1, n+1):
                indexing.append((matchL[i], i))  # (s_i, index i)
            indexing.sort()  # ascending by s_i

            # 5) Try to allocate decrements.
            # free_slots[s] = True means time s is not used for marking
            free_slots = [True]*(T+1)
            for s_i, i in indexing:
                free_slots[s_i] = False

            # prefix list of free times to make it easy to find
            free_times = []
            for s in range(1, T+1):
                if free_slots[s]:
                    free_times.append(s)

            # We'll allocate nums[i] from the free_times that are strictly less than s_i.
            # So let's do a pointer approach or binary search approach to find how many free slots < s_i.
            # Then we must remove those from free_times as we use them.
            # The simplest is an index pointer over free_times as we go from smallest s_i to largest.
            # For each i, we need to pick exactly nums[i] slots from free_times that are < s_i.

            # We'll do it with a small index pointer inside free_times.
            # We can do a pass for each (s_i, i) in ascending s_i. 
            # We'll try to find nums[i] times < s_i from the "unused" portion of free_times.
            # If we can't find enough, fail.

            # Implementation detail: we can do a two-pointer approach:
            # We maintain a pointer "ft_ptr" into free_times that scans left->right.
            # But for each i, we need to pick times < s_i. We can do a loop while ft_ptr < len(free_times)
            # and free_times[ft_ptr] < s_i. If we can pick nums[i] of them, we do so. Otherwise fail.
            # We'll increment ft_ptr accordingly.
            ft_ptr = 0
            for s_i, i in indexing:
                needed = nums[i-1]
                # we must pick "needed" free times < s_i
                count_used = 0
                while count_used < needed and ft_ptr < len(free_times) and free_times[ft_ptr] < s_i:
                    # use free_times[ft_ptr]
                    ft_ptr += 1
                    count_used += 1

                if count_used < needed:
                    # not enough free slots < s_i
                    return False

            return True

        # We'll do a binary search for the smallest T in [total_decrements + n .. m] 
        # for which can_feasibly_mark_all(T) is True.
        # If none, return -1.

        left = total_decrements + n
        if left > m:
            return -1
        right = m

        # Prepare data structures for matching
        matchL = [0]*(n+1)  # which right-side node is matched to i
        matchR = [0]*(m+1)  # which left-side node is matched to s
        dist = [0]*(n+1)    # for BFS in Hopkroft-Karp

        ans = -1
        while left <= right:
            mid = (left + right)//2
            if can_feasibly_mark_all(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans