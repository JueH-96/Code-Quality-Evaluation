class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)

        def can_transform(s1, s2):
            if s1 == s2:
                return True
            n1 = len(s1)
            n2 = len(s2)
            if n1 != n2:
                return False

            for i in range(n1):
                for j in range(i + 1, n1):
                    temp = list(s1)
                    temp[i], temp[j] = temp[j], temp[i]
                    if "".join(temp) == s2:
                        return True
            return False

        def are_almost_equal(num1, num2):
            s1 = str(num1)
            s2 = str(num2)

            if num1 == num2:
                return True

            return can_transform(s1, s2) or can_transform(s2, s1)

        for i in range(n):
            for j in range(i + 1, n):
                if are_almost_equal(nums[i], nums[j]):
                    count += 1
        return count