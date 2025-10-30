class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count_even = 0
        count_odd = 0
        even_seq = 0
        odd_seq = 0
        for num in nums:
            if num % 2 == 0:
                count_even += 1
                even_seq = odd_seq + 1
            else:
                count_odd += 1
                odd_seq = even_seq + 1
        return max(count_even, count_odd, even_seq, odd_seq)