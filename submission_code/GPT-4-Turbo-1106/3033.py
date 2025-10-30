class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # Calculate the number of positions where s1 and s2 differ
        diff_positions = [i for i in range(len(s1)) if s1[i] != s2[i]]
        
        # If the number of differing positions is odd, it's impossible to make them equal
        if len(diff_positions) % 2 != 0:
            return -1
        
        # If x is greater than or equal to 2, we can always use the second operation
        # since it's cheaper or equal to the first operation
        if x >= 2:
            return len(diff_positions) // 2
        
        # If x is less than 2, we need to check if the differing positions are adjacent
        # If they are not adjacent, we will need to use the first operation
        cost = 0
        i = 0
        while i < len(diff_positions):
            # If the next differing position is adjacent, use the second operation
            if i + 1 < len(diff_positions) and diff_positions[i] + 1 == diff_positions[i + 1]:
                cost += 1
                i += 2
            else:
                # If x is 1 and we have a non-adjacent pair, we must use the first operation
                if x == 1:
                    return -1
                cost += x
                i += 2
        
        return cost