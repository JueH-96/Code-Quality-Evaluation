class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1 = max(nums1)
        max2 = max(nums2)

        def calculate_swaps(target_last1, target_last2):
            min_swaps = float('inf')

            # Case 1: No swap at the last index
            if nums1[n - 1] == target_last1 and nums2[n - 1] == target_last2:
                swaps = 0
                possible = True
                temp_nums1 = list(nums1)
                temp_nums2 = list(nums2)
                for i in range(n - 1):
                    if temp_nums1[i] > target_last1 and temp_nums2[i] > target_last2:
                        possible = False
                        break
                    elif temp_nums1[i] > target_last1:
                        swaps += 1
                        temp_nums1[i], temp_nums2[i] = temp_nums2[i], temp_nums1[i]
                    elif temp_nums2[i] > target_last2:
                        swaps += 1
                        temp_nums1[i], temp_nums2[i] = temp_nums2[i], temp_nums1[i]
                if possible:
                    min_swaps = min(min_swaps, swaps)

            # Case 2: Swap at the last index
            if nums1[n - 1] == target_last2 and nums2[n - 1] == target_last1:
                swaps = 1
                possible = True
                temp_nums1 = list(nums2)
                temp_nums2 = list(nums1)
                for i in range(n - 1):
                    if nums1[i] > target_last1 and nums2[i] > target_last2:
                        possible = False
                        break
                    elif nums1[i] > target_last1:
                        swaps += 1
                    elif nums2[i] > target_last2:
                        swaps += 1
                if possible:
                    min_swaps = min(min_swaps, swaps)

            return min_swaps

        ans = float('inf')

        # Target: nums1[-1] = max1, nums2[-1] = max2
        swaps1 = 0
        possible1 = True
        temp_nums1_1 = list(nums1)
        temp_nums2_1 = list(nums2)
        for i in range(n - 1):
            if temp_nums1_1[i] > max1 and temp_nums2_1[i] > max2:
                possible1 = False
                break
            elif temp_nums1_1[i] > max1:
                swaps1 += 1
                temp_nums1_1[i], temp_nums2_1[i] = temp_nums2_1[i], temp_nums1_1[i]
            elif temp_nums2_1[i] > max2:
                swaps1 += 1
                temp_nums1_1[i], temp_nums2_1[i] = temp_nums2_1[i], temp_nums1_1[i]
        if possible1 and temp_nums1_1[n-1] == max1 and temp_nums2_1[n-1] == max2:
            ans = min(ans, swaps1)

        swaps2 = 1
        possible2 = True
        temp_nums1_2 = list(nums2)
        temp_nums2_2 = list(nums1)
        for i in range(n - 1):
            if nums1[i] > max1 and nums2[i] > max2:
                possible2 = False
                break
            elif nums1[i] > max1:
                swaps2 += 1
            elif nums2[i] > max2:
                swaps2 += 1
        if possible2 and nums2[n-1] == max1 and nums1[n-1] == max2:
             ans = min(ans, swaps2)

        swaps3 = 1
        possible3 = True
        temp_nums1_3 = list(nums2)
        temp_nums2_3 = list(nums1)
        for i in range(n - 1):
            if nums1[i] > max2 and nums2[i] > max1:
                possible3 = False
                break
            elif nums1[i] > max2:
                swaps3 += 1
            elif nums2[i] > max1:
                swaps3 += 1

        if possible3 and nums2[n-1] == max2 and nums1[n-1] == max1:
            ans = min(ans, swaps3)

        if ans == float('inf'):
            return -1
        return ans