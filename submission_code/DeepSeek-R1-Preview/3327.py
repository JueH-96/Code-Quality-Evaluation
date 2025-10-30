class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, num in enumerate(nums) if num == 1]
        min_moves = float('inf')
        n = len(nums)
        
        if not ones:
            return 2 * k
        
        for a in range(n):
            s = 1 if nums[a] == 1 else 0
            needed = k - s
            if needed < 0:
                current_moves = 0
            else:
                if s == 1:
                    others = [abs(i - a) for i in ones if i != a]
                else:
                    others = [abs(i - a) for i in ones]
                others.sort()
                m = min(needed, len(others))
                sum_ds = sum(others[:m]) if m > 0 else 0
                over_two = []
                for d in others[:m]:
                    if d > 2:
                        over_two.append(d - 2)
                over_two.sort(reverse=True)
                t = min(maxChanges, len(over_two))
                sum_ds -= sum(over_two[:t])
                needed_remaining = needed - m
                type1_remaining = maxChanges - t
                additional = min(type1_remaining, needed_remaining)
                sum_ds += 2 * additional
                needed_remaining -= additional
                sum_ds += 2 * needed_remaining
                current_moves = sum_ds
            if current_moves < min_moves:
                min_moves = current_moves
        
        return min_moves