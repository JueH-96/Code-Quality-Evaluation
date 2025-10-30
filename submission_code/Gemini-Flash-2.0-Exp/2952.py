class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_sum1 = sum(nums1)
        total_sum2 = sum(nums2)
        
        items = []
        for i in range(n):
            items.append((nums1[i], nums2[i]))
        
        items.sort(key=lambda item: item[1])
        
        nums1_sorted = [item[0] for item in items]
        nums2_sorted = [item[1] for item in items]
        
        dp = {}
        
        def solve(index, time):
            if index == n:
                current_sum = total_sum1 + time * total_sum2
                return current_sum <= x
            
            if (index, time) in dp:
                return dp[(index, time)]
            
            # Option 1: Don't make nums1[index] = 0
            if solve(index + 1, time):
                dp[(index, time)] = True
                return True
            
            # Option 2: Make nums1[index] = 0
            new_total_sum1 = total_sum1 - nums1_sorted[index]
            new_total_sum2 = total_sum2 - nums2_sorted[index]
            
            temp_nums1 = nums1_sorted[:]
            temp_nums2 = nums2_sorted[:]
            
            temp_nums1[index] = 0
            
            
            
            if new_total_sum1 + (time + 1) * new_total_sum2 <= x:
                dp[(index, time)] = True
                return True
            
            dp[(index, time)] = False
            return False
        
        for time in range(n + 1):
            if total_sum1 + time * total_sum2 <= x:
                return time
            
            dp = {}
            if solve(0, time):
                return time
        
        return -1