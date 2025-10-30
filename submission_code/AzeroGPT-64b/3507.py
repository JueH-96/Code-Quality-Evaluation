class Solution:
    def isNotSpecial(self, x):
        if x == 1:
            return True
        cnt = 0
        for k in range(2, int(x**0.5)+1):
            if x % k == 0:
                if x//k == k:
                    return True
                else:
                    return False
        return True

    def nonSpecialCount(self, l: int, r: int) -> int:
        return sum([self.isNotSpecial(i) for i in range(l, r+1)])