class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n < 3:
            return []
        
        # Precompute the is_peak array
        is_peak = [0] * n
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                is_peak[i] = 1
            else:
                is_peak[i] = 0
        
        # Fenwick Tree implementation
        class FenwickTree:
            def __init__(self, size):
                self.n = size
                self.tree = [0] * (self.n + 1)  # 1-based indexing
            
            def update(self, index, delta):
                # Update the Fenwick tree at the given index (1-based) by delta
                while index <= self.n:
                    self.tree[index] += delta
                    index += index & -index
            
            def query(self, index):
                # Query the prefix sum up to the given index (1-based)
                res = 0
                while index > 0:
                    res += self.tree[index]
                    index -= index & -index
                return res
        
        # Initialize the Fenwick Tree
        ft = FenwickTree(n)
        for i in range(n):
            ft.update(i + 1, is_peak[i])  # i+1 is the 1-based index in the Fenwick Tree
        
        answer = []
        
        for query in queries:
            if query[0] == 1:
                # Type 1: Count peaks in [l, r]
                l = query[1]
                r = query[2]
                if r - l < 2:
                    answer.append(0)
                    continue
                a = l + 1
                b = r - 1
                if a > b:
                    answer.append(0)
                    continue
                sum_b = ft.query(b + 1)  # sum from 1..b+1 (0-based b)
                sum_a_minus_1 = ft.query(a)
                result = sum_b - sum_a_minus_1
                answer.append(result)
            else:
                # Type 2: Update nums[index] to val
                index = query[1]
                val = query[2]
                nums[index] = val
                # Check the peak status for index-1, index, index+1
                for dj in (-1, 0, 1):
                    j = index + dj
                    if j < 1 or j >= n - 1:
                        continue  # j must be in [1, n-2] to be a possible peak
                    # Calculate new_val
                    if nums[j] > nums[j - 1] and nums[j] > nums[j + 1]:
                        new_val = 1
                    else:
                        new_val = 0
                    if new_val != is_peak[j]:
                        delta = new_val - is_peak[j]
                        ft.update(j + 1, delta)  # j+1 is the 1-based index
                        is_peak[j] = new_val
        
        return answer