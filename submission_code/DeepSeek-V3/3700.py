from collections import defaultdict
from math import comb

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n < 5:
            return 0
        
        # Precompute the frequency of each number
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        result = 0
        
        # Iterate over each possible middle element
        for mid in freq:
            # The middle element must appear at least once
            if freq[mid] < 1:
                continue
            # The middle element must be the mode and unique
            # So, in the subsequence, the middle element must appear more than any other element
            # And no other element should appear as many times as the middle element
            # Since the subsequence is of size 5, the middle element must appear at least once
            # and at most 3 times (since if it appears 4 or 5 times, it's the only mode)
            # But to have a unique mode, the middle element must appear more than any other element in the subsequence
            # So, the middle element must appear at least once, and the other elements must appear less than the middle element's count in the subsequence
            # To simplify, we can consider that in the subsequence, the middle element appears at least once, and the other elements appear at most once
            # Because if any other element appears more than once, it could potentially have the same frequency as the middle element, making the mode not unique
            # So, for the subsequence to have a unique middle mode, the middle element must appear at least once, and the other four elements must be distinct and different from the middle element
            # So, the subsequence must be of the form [a, b, mid, c, d], where a, b, c, d are distinct and not equal to mid
            # So, we need to count the number of ways to choose 4 distinct elements from the remaining elements (excluding mid) and arrange them around mid
            # The number of ways to choose 4 distinct elements from the remaining elements is C(total_remaining, 4)
            # And for each such choice, there are 4! ways to arrange them around mid
            # So, the total number of such subsequences is C(total_remaining, 4) * 4!
            # Where total_remaining is the count of elements not equal to mid
            total_remaining = n - freq[mid]
            if total_remaining < 4:
                continue
            # Calculate the number of ways to choose 4 distinct elements from the remaining elements
            # Since the elements are not necessarily unique, we need to consider the frequency of each element
            # But to ensure that the four elements are distinct, we need to choose 4 distinct elements from the unique elements in the remaining list
            # So, first, count the number of unique elements in the remaining list
            unique_remaining = 0
            for num in freq:
                if num != mid:
                    unique_remaining += 1
            if unique_remaining < 4:
                continue
            # The number of ways to choose 4 distinct elements from the unique_remaining elements is C(unique_remaining, 4)
            ways_to_choose = comb(unique_remaining, 4)
            # For each chosen 4 elements, there are 4! ways to arrange them around mid
            ways_to_arrange = 24  # 4! = 24
            # The total number of such subsequences is ways_to_choose * ways_to_arrange
            total_subseq = ways_to_choose * ways_to_arrange
            # Multiply by the number of ways to choose the middle element (since it can appear multiple times)
            # The middle element can appear once, twice, or three times in the subsequence
            # But to have a unique mode, the middle element must appear more than any other element in the subsequence
            # So, if the middle element appears k times, the other elements must appear less than k times
            # Since the subsequence is of size 5, the middle element can appear at most 3 times (since if it appears 4 or 5 times, it's the only mode)
            # So, we need to consider the cases where the middle element appears 1, 2, or 3 times
            # For each case, we need to ensure that the other elements appear less than the middle element's count
            # So, for the middle element appearing k times, the other elements must appear at most k-1 times
            # But since the subsequence is of size 5, and the middle element appears k times, the other elements must appear at most (5 - k) / 4 times
            # Wait, perhaps it's easier to consider that the middle element appears exactly once, and the other elements appear at most once
            # Because if the middle element appears more than once, it's harder to ensure that no other element appears as many times
            # So, for simplicity, let's consider that the middle element appears exactly once, and the other elements appear at most once
            # So, the subsequence is of the form [a, b, mid, c, d], where a, b, c, d are distinct and not equal to mid
            # So, the number of such subsequences is C(total_remaining, 4) * 4! * C(freq[mid], 1)
            # Because we choose 4 distinct elements from the remaining, arrange them around mid, and choose one occurrence of mid
            # So, the total number of such subsequences is ways_to_choose * ways_to_arrange * freq[mid]
            total_subseq = ways_to_choose * ways_to_arrange * freq[mid]
            result += total_subseq
            result %= MOD
        
        return result