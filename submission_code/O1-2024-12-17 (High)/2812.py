class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # From the problem statement and examples, it can be deduced that for x to
        # become equal to num within t operations, the largest possible x is:
        #
        #    x = num + 2 * t
        #
        # Explanation:
        # Each operation can change the difference between x and num by 2
        # (by incrementing one while decrementing the other). Therefore,
        # the maximum initial difference we can neutralize in t moves is 2*t.
        # Hence, starting from x = num + 2*t and performing t steps of 
        # (x-1, num+1) will end at x = num, both meeting at num + t finally.
        #
        return num + 2 * t