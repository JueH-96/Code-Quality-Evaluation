class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        # First, collect all positions where s1 and s2 differ.
        mismatches = []
        for i in range(n):
            if s1[i] != s2[i]:
                mismatches.append(i)
                
        m = len(mismatches)
        # Each allowed operation flips two bits.
        # Hence if the number of mismatches is odd it is impossible.
        if m % 2 == 1:
            return -1
        
        # The key observation is as follows:
        # To fix a mismatch, you must flip it an odd number of times.
        # Since each operation always flips exactly two bits, one may “pair if up” the
        # mismatched positions so that each operation fixes exactly two bits.
        #
        # There are two ways to flip a pair:
        # 1. Use the arbitrary pair flip (choose any two indices) at cost x.
        # 2. Use a sequence of adjacent flips (each flipping i and i+1 at cost 1) to "move"
        #    one of the mismatches until it is adjacent to the other and then fix them
        #    with a single adjacent flip.
        #
        # It turns out that if you have a pair of mismatches at positions i and j (with i < j)
        # then you can fix them with cost = min( x, (j-i) ).
        # (For example, if they are consecutive then an adjacent op costs 1,
        #  otherwise you might “slide” the error at a cost equal to the gap.)
        #
        # A classical fact for perfect matching on a line with a cost that depends monotonically
        # on the gap is that pairing them in their natural order (first with second, third with fourth, …)
        # is optimal.
        #
        # Therefore, we simply take the mismatches in order and for each consecutive pair add:
        #      cost += min( x, (mismatches[i+1] - mismatches[i]) )
        
        total_cost = 0
        i = 0
        while i < m:
            total_cost += min(x, mismatches[i+1] - mismatches[i])
            i += 2
            
        return total_cost

# ---------------------------
# Example test cases
# ---------------------------
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    s1 = "1100011000"
    s2 = "0101001010"
    x = 2
    # Explanation: One optimal sequence is:
    #   - Use an adjacent flip at index 3 (cost 1)
    #   - Use an adjacent flip at index 4 (cost 1)
    #   - Use an arbitrary flip on indices 0 and 8 (cost 2)
    # Total cost = 1 + 1 + 2 = 4.
    print(sol.minOperations(s1, s2, x))  # Expected output: 4

    # Example 2:
    s1 = "10110"
    s2 = "00011"
    x = 4
    # Explanation: Mismatches occur in an odd number of indices so it is impossible.
    print(sol.minOperations(s1, s2, x))  # Expected output: -1