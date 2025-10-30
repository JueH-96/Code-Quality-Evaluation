class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # The key idea is to look at the positions where s1 and s2 differ.
        # Denote by diff the list of indices i so that s1[i] != s2[i].
        # Notice that both kinds of operations flip exactly two bits.
        # In an operation we always flip an even number (2) of bits.
        # Consequently, if the total number of mismatches is odd, it is impossible to fix all differences.
        #
        # There are two kinds of operations:
        # 1. Arbitrary pair flip: Choose any two indices and flip them at cost x.
        #    This operation can “cancel” any two mismatches, regardless of their positions.
        #
        # 2. Adjacent flip: Choose an index i (with i < n-1) and flip bits i and (i+1) at cost 1.
        #    When both bits are mismatches then this operation fixes them.
        #    If only one is a mismatch, it “moves” the error to the adjacent position.
        #
        # Thus for any two mismatches (located at indices i and j, with i < j) 
        # we can “cancel” them in one of two ways:
        #   A. Use arbitrary pair flip at cost = x.
        #   B. If we want to merge them via adjacent moves, then we can “slide” one error toward the other.
        #      By repeated adjacent operations we can effectively cancel them at cost = (j - i) (because it costs 1 per adjacent move,
        #      and when they become consecutive a single adjacent op will cancel both).
        #
        # So for any pair (i, j), the cost to clear them is:
        #       cost_pair = min(x, j - i)
        #
        # An important special situation arises when there are exactly 2 mismatches AND they are consecutive.
        # In that case the cost by adjacent operation is 1 (not 2).
        # (For example, s1 = "01", s2 = "10": diff indices 0 and 1.
        #  An adjacent flip on (0,1) turns "01" into "10" for a cost 1;
        #  while the arbitrary op would cost x.)
        #
        # Therefore the plan is:
        # 1. Compute diff: the list of indices where s1 and s2 differ.
        # 2. If len(diff) is odd, return -1.
        # 3. If len(diff)==2 and they are consecutive, return min(x, 1).
        # 4. Otherwise, pair the mismatches in order (which is optimal)
        #    and sum the cost for each pair as min(x, diff[i+1]-diff[i]).
        
        n = len(s1)
        diff = []
        for i in range(n):
            if s1[i] != s2[i]:
                diff.append(i)
                
        m = len(diff)
        if m % 2 == 1:
            return -1
        if m == 0:
            return 0
        
        # Special case: exactly two mismatches and they are consecutive
        if m == 2 and diff[1] - diff[0] == 1:
            return min(x, 1)
            
        cost = 0
        i = 0
        while i < m:
            gap = diff[i+1] - diff[i]
            cost += min(x, gap)
            i += 2
        return cost

# For basic testing:
if __name__ == "__main__":
    sol = Solution()

    # Example 1:
    s1 = "1100011000"
    s2 = "0101001010"
    x = 2
    # Mismatches are at positions: 0, 3, 5, 8.
    # We can pair (0,3) at cost = min(2, 3)=2 and (5,8) at cost = min(2, 3)=2.
    # Total cost = 2+2 = 4.
    print(sol.minOperations(s1, s2, x))  # Expected output: 4

    # Example 2:
    s1 = "10110"
    s2 = "00011"
    x = 4
    # The number of mismatches is odd so it is impossible.
    print(sol.minOperations(s1, s2, x))  # Expected output: -1