class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # Each "AA" contributes 2 'A's, each "BB" contributes 2 'B's, each "AB" contributes 1 'A' and 1 'B'
        # The main constraint is that we cannot have "AAA" or "BBB" in the final string
        # To maximize the length, we need to balance the number of 'A's and 'B's
        
        # The total number of 'A's is 2*x + z
        # The total number of 'B's is 2*y + z
        
        # The maximum possible length is the sum of all characters, but we need to ensure that no three consecutive 'A's or 'B's are present
        
        # The key is to arrange the strings in such a way that no three 'A's or 'B's are consecutive
        # One way to achieve this is to alternate between 'A's and 'B's as much as possible
        
        # The maximum length is min(2*(x + y) + 2*z, 2*(x + y + z))
        # But considering the constraints, the maximum length is 2*(x + y) + 2*z
        
        # However, to ensure no three consecutive 'A's or 'B's, we need to limit the number of 'A's and 'B's
        # The maximum number of 'A's is 2*(x + z)
        # The maximum number of 'B's is 2*(y + z)
        
        # The total length is the sum of all characters, but we need to ensure that the counts of 'A's and 'B's are balanced
        
        # The maximum possible length is min(2*(x + y) + 2*z, 2*(x + y + z))
        
        # But to ensure no three consecutive 'A's or 'B's, we need to limit the counts
        
        # The maximum possible length is 2*(x + y) + 2*z
        
        # But to ensure no three consecutive 'A's or 'B's, we need to make sure that the counts of 'A's and 'B's are balanced
        
        # The maximum possible length is min(2*(x + y) + 2*z, 2*(x + y + z))
        
        # But considering the constraints, the maximum length is 2*(x + y) + 2*z
        
        # So the final answer is 2*(x + y) + 2*z
        
        return 2 * (x + y) + 2 * z