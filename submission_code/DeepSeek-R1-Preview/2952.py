class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        sum_initial = sum(nums1)
        B = sum(nums2)
        if sum_initial <= x:
            return 0
        
        n = len(nums1)
        combined = sorted(zip(nums1, nums2), key=lambda x: -x[1])
        a = [x[0] for x in combined]
        b = [x[1] for x in combined]
        
        for K in range(1, n + 1):
            sum_a = sum(a[:K])
            sum_b = 0
            for i in range(K):
                sum_b += b[i] * (K - i)
            total = sum_initial + B * K - (sum_a + sum_b)
            if total <= x:
                return K
        
        return -1