from typing import List
from collections import Counter
import heapq

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        nums1_counter = Counter(nums1)
        nums2_counter = Counter(nums2)
        nums1_counter_heap = [(-count, num) for num, count in nums1_counter.items()]
        nums2_counter_heap = [(-count, num) for num, count in nums2_counter.items()]
        heapq.heapify(nums1_counter_heap)
        heapq.heapify(nums2_counter_heap)
        set_size = 0
        while nums1_counter_heap and nums2_counter_heap:
            count1, num1 = heapq.heappop(nums1_counter_heap)
            count2, num2 = heapq.heappop(nums2_counter_heap)
            if num1 == num2:
                set_size += 1
                if count1 < -1:
                    heapq.heappush(nums1_counter_heap, (count1 + 1, num1))
                if count2 < -1:
                    heapq.heappush(nums2_counter_heap, (count2 + 1, num2))
            else:
                if count1 < -1:
                    heapq.heappush(nums1_counter_heap, (count1 + 1, num1))
                if count2 < -1:
                    heapq.heappush(nums2_counter_heap, (count2 + 1, num2))
        return set_size