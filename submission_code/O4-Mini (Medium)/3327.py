from bisect import bisect_left
class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        # Precompute positions of ones and zeros
        ones = [i for i, v in enumerate(nums) if v == 1]
        zeros = [i for i, v in enumerate(nums) if v == 0]
        # Helper: can we achieve k pickups in <= moves?
        def can(moves: int) -> bool:
            # for each possible starting index i
            for i in range(n):
                # pointers into ones and zeros lists
                lo1 = bisect_left(ones, i)
                lo0 = bisect_left(zeros, i)
                l1 = lo1 - 1
                r1 = lo1
                l0 = lo0 - 1
                r0 = lo0
                taken = 0
                used_zeros = 0
                total = 0
                # greedy pick k sources by lowest cost
                while taken < k:
                    # pick next best among up to four candidates
                    best = None
                    cost = None
                    typ = None
                    # candidate from left-one
                    if l1 >= 0:
                        d = i - ones[l1]
                        if d <= moves:
                            c = d
                            best = ('one', 'L')
                            cost = c
                    # right-one
                    if r1 < len(ones):
                        d = ones[r1] - i
                        if d <= moves:
                            c = d
                            if best is None or c < cost:
                                best = ('one', 'R'); cost = c
                    # left-zero
                    if used_zeros < maxChanges and l0 >= 0:
                        d = i - zeros[l0]
                        if d + 1 <= moves:
                            c = d + 1
                            if best is None or c < cost:
                                best = ('zero', 'L'); cost = c
                    # right-zero
                    if used_zeros < maxChanges and r0 < len(zeros):
                        d = zeros[r0] - i
                        if d + 1 <= moves:
                            c = d + 1
                            if best is None or c < cost:
                                best = ('zero', 'R'); cost = c
                    if best is None:
                        break
                    # take it
                    if best[0] == 'one':
                        if best[1] == 'L': l1 -= 1
                        else: r1 += 1
                    else:
                        used_zeros += 1
                        if best[1] == 'L': l0 -= 1
                        else: r0 += 1
                    total += cost
                    if total > moves:
                        break
                    taken += 1
                if taken == k and total <= moves:
                    return True
            return False

        # binary search answer
        # minimal moves is at least (k-1) because initial pick costs 0, each other needs at least one move
        left, right = 0, n + k + 5
        # tighten left bound
        left = max(0, k - 1)
        ans = right
        while left <= right:
            mid = (left + right) // 2
            if can(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans