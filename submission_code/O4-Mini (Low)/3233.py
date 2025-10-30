class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        from functools import lru_cache
        
        # compute end index for partition starting at i
        def compute_end(i):
            seen = set()
            for j in range(i, n):
                seen.add(s[j])
                if len(seen) > k:
                    return j - 1, True  # end at j-1, and we hit k+1 at j
            return n - 1, False  # consumed to end, never hit k+1
        
        # memoized rest partitions from i
        from functools import lru_cache
        @lru_cache(None)
        def rest_parts(i):
            if i >= n:
                return 0
            e, _ = compute_end(i)
            return 1 + rest_parts(e + 1)
        
        # simulate original to get partition starts and count m
        starts = []
        hit_extra = []  # did this partition hit k+1?
        i = 0
        while i < n:
            e, extra = compute_end(i)
            starts.append(i)
            hit_extra.append(extra)
            i = e + 1
        
        total_orig = rest_parts(0)
        ans = total_orig  # no-change case
        
        # for each partition start where we can introduce new distinct
        for m, i in enumerate(starts):
            if not hit_extra[m]:
                continue  # never hit k+1, can't shorten
            # if we change s[i] to that extra letter, this partition ends at i
            # gives one small partition + rest from i+1
            new_total = m + 1 + rest_parts(i + 1)
            if new_total > ans:
                ans = new_total
        
        return ans