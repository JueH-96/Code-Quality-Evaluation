class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def count_digits(n):
            return len(str(n))

        def can_swap(n1, n2):
            if n1 == n2:
                return True
            diff_digits = [(d1, d2) for d1, d2 in zip(str(n1), str(n2)) if d1 != d2]
            return len(diff_digits) == 2 and diff_digits[0] in diff_digits[1:]

        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if can_swap(nums[i], nums[j]):
                    count += 1
        return count