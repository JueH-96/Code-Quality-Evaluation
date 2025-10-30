class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Precompute the positions for each index
        pre = [[] for _ in range(n)]
        for s in range(m):
            val = changeIndices[s]
            i = val - 1  # convert to 0-based index
            if 0 <= i < n:
                pre[i].append(s)
        
        # Check if any index has no occurrence in changeIndices
        for i in range(n):
            if not pre[i]:
                return -1
        
        # Binary search setup
        low = 1
        high = m
        ans = -1
        
        sum_nums = sum(nums)
        required = sum_nums + n
        
        while low <= high:
            mid = (low + high) // 2
            
            # Check if required operations exceed mid
            if required > mid:
                low = mid + 1
                continue
            
            valid = True
            for i in range(n):
                pos_list = pre[i]
                # Binary search for the largest position <= mid-1
                left, right = 0, len(pos_list) - 1
                best = -1
                while left <= right:
                    m_pos = (left + right) // 2
                    if pos_list[m_pos] <= mid - 1:
                        best = m_pos
                        left = m_pos + 1
                    else:
                        right = m_pos - 1
                if best == -1:
                    valid = False
                    break
                s_i = pos_list[best] + 1  # convert to 1-based
                if s_i < nums[i] + 1:
                    valid = False
                    break
            
            if valid:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans