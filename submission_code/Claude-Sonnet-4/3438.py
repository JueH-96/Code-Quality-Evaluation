class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        
        # Binary Indexed Tree for range sum queries
        class BIT:
            def __init__(self, size):
                self.size = size
                self.tree = [0] * (size + 1)
            
            def update(self, i, delta):
                i += 1  # 1-indexed
                while i <= self.size:
                    self.tree[i] += delta
                    i += i & (-i)
            
            def query(self, i):
                i += 1  # 1-indexed
                result = 0
                while i > 0:
                    result += self.tree[i]
                    i -= i & (-i)
                return result
            
            def range_query(self, l, r):
                if l > r:
                    return 0
                return self.query(r) - (self.query(l - 1) if l > 0 else 0)
        
        bit = BIT(n)
        
        # Check if index i is a peak
        def is_peak(i):
            if i <= 0 or i >= n - 1:
                return False
            return nums[i] > nums[i - 1] and nums[i] > nums[i + 1]
        
        # Initialize BIT with current peaks
        for i in range(1, n - 1):
            if is_peak(i):
                bit.update(i, 1)
        
        result = []
        
        for query in queries:
            if query[0] == 1:  # Count peaks in range
                l, r = query[1], query[2]
                # We can only have peaks from index l+1 to r-1 (inclusive)
                # because first and last elements of subarray cannot be peaks
                if l + 1 <= r - 1:
                    count = bit.range_query(l + 1, r - 1)
                else:
                    count = 0
                result.append(count)
            
            else:  # Update element
                index, val = query[1], query[2]
                
                # Check which positions might be affected (index-1, index, index+1)
                affected_positions = []
                for pos in [index - 1, index, index + 1]:
                    if 1 <= pos <= n - 2:  # Only positions that can be peaks
                        affected_positions.append(pos)
                
                # Remove old peaks
                for pos in affected_positions:
                    if is_peak(pos):
                        bit.update(pos, -1)
                
                # Update the value
                nums[index] = val
                
                # Add new peaks
                for pos in affected_positions:
                    if is_peak(pos):
                        bit.update(pos, 1)
        
        return result