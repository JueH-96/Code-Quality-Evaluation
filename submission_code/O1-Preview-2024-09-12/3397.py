class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        from collections import Counter
        counter_nums1 = Counter(nums1)
        counter_nums2 = Counter(nums2)
        possible_xs = set()
        for num2 in counter_nums2:
            for num1 in counter_nums1:
                x = num2 - num1
                possible_xs.add(x)
        for x in possible_xs:
            match = True
            for num2 in counter_nums2:
                if counter_nums2[num2] != counter_nums1.get(num2 - x, 0):
                    match = False
                    break
            if match:
                return x
        return 0  # Since an integer x always exists per constraints