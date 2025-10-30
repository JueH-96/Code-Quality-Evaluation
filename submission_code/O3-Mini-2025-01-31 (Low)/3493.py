class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        ones = []
        for i, ch in enumerate(s):
            if ch == '1':
                ones.append(i)
        k = len(ones)
        # If there is no operation possible (no 1 or string already grouped) return 0.
        if k == 0:
            return 0
        # In the bubble–sort picture, the final positions (target positions)
        # of the ones must be the last k indices: n-k, n-k+1, …, n-1.
        total_swaps = 0
        for index, pos in enumerate(ones):
            target = n - k + index  # desired final position for this '1'
            total_swaps += (target - pos)
        # By an exchange argument, when k>1, it is always optimal to simulate these
        # moves in a “step–by–step” manner (each of cost 1) except that one move is forced 
        # to be “combined” (we lose one distinct move) compared to the adjacent–swap picture.
        # (When k==1, the single one may be moved one–swap at a time.)
        if k > 1:
            return total_swaps - 1
        else:
            return total_swaps