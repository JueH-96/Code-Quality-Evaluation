class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        # Greedy partition count function
        def greedy_count(arr):
            n = len(arr)
            cnt = {}
            partitions = 0
            i = 0
            while i < n:
                cnt.clear()
                distinct = 0
                j = i
                # extend as far as at most k distinct
                while j < n:
                    c = arr[j]
                    if cnt.get(c, 0) == 0:
                        if distinct == k:
                            break
                        distinct += 1
                    cnt[c] = cnt.get(c, 0) + 1
                    j += 1
                partitions += 1
                i = j
            return partitions

        n = len(s)
        # First, compute the greedy block starts on original s
        # We'll get the start indices of each block
        starts = []
        i = 0
        s_arr = list(s)
        while i < n:
            starts.append(i)
            cnt = {}
            distinct = 0
            j = i
            while j < n:
                c = s_arr[j]
                if cnt.get(c, 0) == 0:
                    if distinct == k:
                        break
                    distinct += 1
                cnt[c] = cnt.get(c, 0) + 1
                j += 1
            i = j

        # Collect candidate change positions: for each block start,
        # find the earliest p where distinct(s[i:p]) == k, and p < block_end
        candidates = []
        for st in starts:
            cnt = {}
            distinct = 0
            # we need block_end
            be = st
            # find block end
            while be < n:
                c = s_arr[be]
                if cnt.get(c, 0) == 0:
                    if distinct == k:
                        break
                    distinct += 1
                cnt[c] = cnt.get(c, 0) + 1
                be += 1
            # find p0: earliest position where prefix distinct hits exactly k
            cnt2 = {}
            distinct2 = 0
            p0 = None
            j = st
            while j < n:
                c = s_arr[j]
                if cnt2.get(c, 0) == 0:
                    distinct2 += 1
                cnt2[c] = cnt2.get(c, 0) + 1
                j += 1
                if distinct2 == k:
                    p0 = j  # s[st:j] has k distinct, so p0=j
                    break
            # if p0 valid and strictly less than be, we can shorten
            if p0 is not None and p0 < be:
                candidates.append(p0)
        # deduplicate
        candidates = list(dict.fromkeys(candidates))

        # Evaluate no-change scenario
        best = greedy_count(s_arr)
        # If already maximum possible, return
        if best == n:
            return best

        # For each candidate p, simulate change at p with a new dummy char '#'
        # and greedy count
        for p in candidates:
            # we want a char not in the block's prefix distinct set;
            # but using '#' works to always be new
            arr2 = s_arr[:]  # copy
            arr2[p] = '#'    # dummy new letter
            cntp = greedy_count(arr2)
            if cntp > best:
                best = cntp
                if best == n:
                    break

        return best