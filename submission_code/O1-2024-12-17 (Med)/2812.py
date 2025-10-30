class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Each operation can change the difference between x and num by -2, 0, or +2.
        # To maximize x while still allowing x to become equal to num in at most t operations,
        # we can set x = num + 2*t. We can then reduce the difference (which is 2*t) to 0 
        # in exactly t steps by applying "decrease x by 1" and "increase num by 1" each time.
        return num + 2*t