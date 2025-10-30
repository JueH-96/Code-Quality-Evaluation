class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        count = 0
        for split1 in range(1, n):
            for split2 in range(split1 + 1, n):
                nums1 = nums[0:split1]
                nums2 = nums[split1:split2]
                nums3 = nums[split2:n]
                
                def is_prefix(arr1, arr2):
                    if len(arr1) > len(arr2):
                        return False
                    for i in range(len(arr1)):
                        if arr1[i] != arr2[i]:
                            return False
                    return True
                    
                if is_prefix(nums1, nums2) or is_prefix(nums2, nums3):
                    count += 1
        return count