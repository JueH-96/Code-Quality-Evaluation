class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # For each operation, we have:
        #   x becomes x + a and num becomes num + b, where a and b are either +1 or -1.
        # After t operations, if we denote A = sum(a_i) and B = sum(b_i),
        # then the final values are:
        #   x_final = x + A, and num_final = num + B.
        # We require x_final to equal num_final.
        # This gives: x + A = num + B  ->  x = num + (B - A)
        #
        # Our goal is to maximize the achievable x.
        # Notice that in each operation, a and b are chosen independently.
        # The net difference (B - A) is the sum over t operations of (b_i - a_i),
        # where each term is one of: (1 - 1)=0, (1 - (-1)) = 2, (-1 - 1) = -2, (-1 - (-1)) = 0.
        # Thus the difference B - A is always an even integer, and its maximum value
        # is 2 * t (by choosing in every operation b = 1 and a = -1).
        #
        # Therefore, the maximum achievable x is:
        #   x_max = num + 2*t
        #
        # This solution is valid because num, t >= 1.
        
        return num + 2 * t