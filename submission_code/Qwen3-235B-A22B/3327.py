from bisect import bisect_left, bisect_right

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, num in enumerate(nums) if num == 1]
        if not ones:
            # All zeros, must create all k ones
            return 2 * k
        
        m = len(ones)
        min_ans = float('inf')

        # Option 1: Use positions in 'ones'
        for alice in ones:
            cnt_initial = 1
            needed = k - cnt_initial
            if needed <= 0:
                min_ans = 0
                continue

            available_dists = []
            for x in ones:
                if x != alice:
                    available_dists.append(abs(x - alice))
            available_dists.sort()
            na = len(available_dists)

            prefix = [0]
            for d in available_dists:
                prefix.append(prefix[-1] + d)

            t_min = max(0, needed - maxChanges)
            t_max = min(na, needed)
            if t_min > t_max:
                continue
            current_cost = float('inf')
            for t in range(t_min, t_max + 1):
                cost = prefix[t] + 2 * (needed - t)
                if cost < current_cost:
                    current_cost = cost
            if current_cost < min_ans:
                min_ans = current_cost

        # Option 2: Consider positions not in 'ones' but near clusters
        # Consider alice positions at x-1, x+1 for each x in ones
        candidates = set()
        for x in ones:
            candidates.add(x - 1)
            candidates.add(x + 1)
        candidates.add(ones[0] - 2)  # Some more candidates
        candidates.add(ones[-1] + 2)
        for x in ones[:min(2, len(ones))]:
            candidates.add(x)
        for x in ones[-min(2, len(ones)):]:
            candidates.add(x)

        for alice in candidates:
            if alice < 0 or alice >= len(nums):
                continue
            # Skip if already processed as part of 'ones'
            if nums[alice] == 1:
                continue
            cnt_initial = 0
            needed = k
            if needed <= 0:
                min_ans = 0
                continue

            available_dists = [abs(x - alice) for x in ones]
            available_dists.sort()
            na = len(available_dists)

            prefix = [0]
            for d in available_dists:
                prefix.append(prefix[-1] + d)

            t_min = max(0, needed - maxChanges)
            t_max = min(na, needed)
            if t_min > t_max:
                continue
            current_cost = float('inf')
            for t in range(t_min, t_max + 1):
                cost = prefix[t] + 2 * (needed - t)
                if cost < current_cost:
                    current_cost = cost
            if current_cost < min_ans:
                min_ans = current_cost

        # Option 3: Using all created ones if possible
        if maxChanges >= k:
            min_ans = min(min_ans, 2 * k)

        return min_ans