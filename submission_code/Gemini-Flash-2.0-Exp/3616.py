class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0

        for start_index in range(n):
            if nums[start_index] == 0:
                # Try moving left
                temp_nums1 = nums[:]
                curr1 = start_index
                direction1 = -1  # -1 for left, 1 for right
                valid1 = True
                while 0 <= curr1 < n:
                    if temp_nums1[curr1] == 0:
                        curr1 += direction1
                    else:
                        temp_nums1[curr1] -= 1
                        direction1 *= -1
                        curr1 += direction1

                if all(x == 0 for x in temp_nums1):
                    count += 1

                # Try moving right
                temp_nums2 = nums[:]
                curr2 = start_index
                direction2 = 1  # -1 for left, 1 for right
                valid2 = True
                while 0 <= curr2 < n:
                    if temp_nums2[curr2] == 0:
                        curr2 += direction2
                    else:
                        temp_nums2[curr2] -= 1
                        direction2 *= -1
                        curr2 += direction2

                if all(x == 0 for x in temp_nums2):
                    count += 1
        return count