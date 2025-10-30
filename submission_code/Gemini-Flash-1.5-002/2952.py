class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        current_sum = sum(nums1)
        if current_sum <= x:
            return 0

        min_time = float('inf')
        for i in range(1 << n):
            temp_nums1 = list(nums1)
            time = 0
            current_sum = sum(temp_nums1)
            
            
            indices_to_zero = []
            for j in range(n):
                if (i >> j) & 1:
                    indices_to_zero.append(j)

            
            while current_sum > x:
                time +=1
                
                
                next_nums1 = []
                for k in range(n):
                    next_nums1.append(temp_nums1[k] + nums2[k])
                
                
                
                if len(indices_to_zero)>0:
                    
                    index_to_zero = indices_to_zero.pop(0)
                    next_nums1[index_to_zero] = 0

                temp_nums1 = next_nums1
                current_sum = sum(temp_nums1)
                if time > n*10 :
                    time = float('inf')
                    break

            if current_sum <=x:
                min_time = min(min_time, time)

        if min_time == float('inf'):
            return -1
        else:
            return min_time