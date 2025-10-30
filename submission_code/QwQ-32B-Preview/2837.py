class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # Function to check if S can be expressed as the sum of exactly k powers of two
        def can_express(S, k):
            if S < 0 or k < 0:
                return False
            if S == 0 and k == 0:
                return True
            if S == 0 and k != 0:
                return False
            # Find the largest power of two less than or equal to S
            i = 0
            max_power = 1
            while max_power * 2 <= S:
                max_power *= 2
                i += 1
            # Subtract this power and recurse
            return can_express(S - max_power, k - 1)
        
        # Iterate over possible k
        for k in range(1, 61):
            S = num1 - k * num2
            if can_express(S, k):
                return k
        return -1