import bisect

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = []
        for i, num in enumerate(nums):
            if num == 1:
                ones.append(i)
        n = len(ones)
        min_moves = float('inf')
        
        # Consider starting positions as ones
        for i in range(n):
            if i + k - 1 >= n:
                continue
            current = ones[i]
            total = 0
            for j in range(i + 1, i + k):
                distance = ones[j] - current
                total += min(1, distance)
            if total < min_moves:
                min_moves = total
        
        # Consider starting positions as zeros, changing one to one
        for i in range(n):
            # Find the position where we can change to one, and then collect the ones
            # The initial step is 1, and then collect the next k-1 ones
            # The next k-1 ones must be after this position
            # We can choose to change any zero to one, so the next k-1 ones can be the first k-1 ones after i
            if i + k - 1 >= n:
                continue
            total = 1  # change to one
            for j in range(i + 1, i + k):
                # The next ones are ones[j] which are already ones
                # The distance is ones[j] - current (current is i)
                # But current is the position after change
                # Wait, no: the starting position is i (changed to one)
                # So the starting position is i, and the next ones are j >= i+1
                # The distance is ones[j] - i
                # So the cost is min(1, ones[j] - i)
                distance = ones[j] - i
                total += min(1, distance)
            if total < min_moves:
                min_moves = total
        
        # Also consider starting at a zero that can be changed to one, but the rest are already ones
        # The minimal sum might be achieved by selecting the first k ones and changing the first zero to one
        # So, compute the cost of changing the first zero to one and then collect the first k ones
        # But this requires that there is at least one zero in the array
        if n == 0:
            return 0
        # Find the position of the first zero that can be changed to one
        first_zero = bisect.bisect_left(ones, 0)
        if first_zero < len(nums) and nums[first_zero] == 0:
            # Change first_zero to one, cost 1
            # Then collect all k ones
            # The ones are at ones[0], ones[1], ..., ones[k-1]
            # The starting position is first_zero
            total = 1
            for j in range(k):
                distance = ones[j] - first_zero
                total += min(1, distance)
            if total < min_moves:
                min_moves = total
        
        return min_moves