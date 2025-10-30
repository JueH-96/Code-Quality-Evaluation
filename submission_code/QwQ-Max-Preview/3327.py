import bisect

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        S = [i for i, num in enumerate(nums) if num == 1]
        n = len(nums)
        min_moves = float('inf')
        sorted_S = sorted(S)
        
        for a in range(n):
            initial_count = 1 if nums[a] == 1 else 0
            required_ones = k - initial_count
            if required_ones <= 0:
                min_moves = min(min_moves, 0)
                continue
            
            existing_ones = sorted_S.copy()
            if nums[a] == 1:
                existing_ones.pop(bisect.bisect_left(existing_ones, a))
            
            m_available = len(existing_ones)
            m_used = min(required_ones, m_available)
            required_ones_after = required_ones - m_used
            if required_ones_after > maxChanges:
                continue
            
            sum_existing = 0
            if m_used > 0:
                pos = bisect.bisect_left(existing_ones, a)
                left = pos - 1
                right = pos
                count = 0
                while count < m_used:
                    if left >= 0 and right < len(existing_ones):
                        if (a - existing_ones[left]) <= (existing_ones[right] - a):
                            sum_existing += a - existing_ones[left]
                            left -= 1
                        else:
                            sum_existing += existing_ones[right] - a
                            right += 1
                    elif left >= 0:
                        sum_existing += a - existing_ones[left]
                        left -= 1
                    else:
                        sum_existing += existing_ones[right] - a
                        right += 1
                    count += 1
            
            sum_changes = required_ones_after * 2
            total = sum_existing + sum_changes
            if total < min_moves:
                min_moves = total
        
        return min_moves