class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        # Special-case: k == 1.
        # For k==1 the greedy segmentation groups consecutive equal characters.
        # Original partition count = number of groups.
        # With one change you can "break" one run into individual letters.
        # For a run of length L, instead of contributing 1, you can get L partitions.
        # So best you could do is: (total groups - 1) + (max run length) 
        # (because you only get to change one index – i.e. split one run).
        if k == 1:
            groups = 0
            max_run = 0
            i = 0
            while i < n:
                groups += 1
                start = i
                while i < n and s[i] == s[start]:
                    i += 1
                run_length = i - start
                if run_length > max_run:
                    max_run = run_length
            # Original segmentation is just groups many partitions.
            original = groups
            candidate = (groups - 1) + max_run
            return max(original, candidate)
        
        # For k >= 2, we first precompute the "next pointer" array using a sliding window.
        # next_ptr[i] will be the smallest index j such that the window s[i:j] (j not included)
        # is the longest prefix with at most k distinct letters.
        next_ptr = [0] * n
        freq = {}
        j = 0
        for i in range(n):
            while j < n and (s[j] in freq or len(freq) < k):
                freq[s[j]] = freq.get(s[j], 0) + 1
                j += 1
            next_ptr[i] = j
            # slide the window: remove s[i]
            freq[s[i]] -= 1
            if freq[s[i]] == 0:
                del freq[s[i]]
        
        # Precompute dp[i]: maximum number of partitions of s[i:] (no modification)
        # Using the greedy segmentation rule: dp[i] = 1 + dp[next_ptr[i]]
        dp = [0] * (n + 1)
        dp[n] = 0
        for i in range(n-1, -1, -1):
            dp[i] = 1 + dp[next_ptr[i]]
        original_ans = dp[0]
        
        # Now simulate the greedy segmentation (without modification) to "collect" segments.
        # For each segment we compute:
        #   L: starting index, R = next_ptr[L] is the break point,
        #   p: the earliest index in [L, R) where the distinct-set becomes full (i.e. size k)
        #   S: the set of characters in s[L ... p] (which will be of size k if p is not None)
        segments = []
        i = 0
        while i < n:
            L = i
            R = next_ptr[i]
            curr_set = set()
            p = None
            for j in range(L, R):
                if s[j] not in curr_set:
                    curr_set.add(s[j])
                    if len(curr_set) == k and p is None:
                        p = j
                # Once we've reached k distinct, we keep scanning.
            segments.append((L, R, p, set(curr_set)))  # store a copy of the set
            i = R
        
        # Helper: Given an index i and a forced letter for s[i], compute the 
        # number of partitions for the segment starting at i using the greedy rule
        # (and then using our precomputed dp for the unchanged remainder).
        def forced_segmentation(i: int, forced: str) -> int:
            # Here, we simulate the greedy segmentation on the substring starting at i,
            # treating s[i] as the forced letter (i.e. having that value instead of the original).
            curSet = {forced}
            j = i + 1
            while j < n:
                if s[j] in curSet:
                    j += 1
                else:
                    if len(curSet) < k:
                        curSet.add(s[j])
                        j += 1
                    else:
                        break
            # Partition count for this segment is 1 + partitions from position j
            return 1 + dp[j]
        
        # Now, try one modification.
        # The idea: In one of the segments, after the set becomes full (at index p),
        # we can "force" an early break by modifying the letter at index candidate = p+1.
        # Modify it to some letter not in S so that the greedy rule stops at candidate.
        candidate_max = 0
        # seg_index is the count of segments (from the start) that are completely unchanged.
        # In the modified segmentation, if we apply a modification in a segment,
        # total partitions = (number of segments before) + 1 (for the shortened segment) + forced_segmentation(result_suffix)
        for seg_index, seg in enumerate(segments):
            L, R, p, segSet = seg
            # Only consider segments where we reached full distinct set (p not None)
            # and where there is at least one character after p (i.e. candidate index exists)
            if p is None:
                continue
            candidate_index = p + 1
            if candidate_index >= R:
                continue
            # For our forced break we want to change the character at candidate_index.
            for ch in "abcdefghijklmnopqrstuvwxyz":
                # We want a ch that is NOT already in our current segment's set;
                # also, if ch equals the original letter, that's not a change.
                if ch in segSet:
                    continue
                if ch == s[candidate_index]:
                    continue
                # Compute segmentation count from candidate_index if we force that letter.
                suffix_count = forced_segmentation(candidate_index, ch)
                total_partition = seg_index + 1 + suffix_count
                if total_partition > candidate_max:
                    candidate_max = total_partition
        
        ans = max(original_ans, candidate_max)
        return ans