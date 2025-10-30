class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        
        # Check if it's possible to make the strings equal
        if sum(1 for c1, c2 in zip(s1, s2) if c1 != c2) % 2 != 0:
            return -1
        
        # Initialize the minimum cost to infinity
        min_cost = float('inf')
        
        # Try all possible pairs of indices for the first operation
        for i in range(n):
            for j in range(i + 1, n):
                cost = x
                s1_list = list(s1)
                
                # Flip the characters at indices i and j
                s1_list[i] = '0' if s1_list[i] == '1' else '1'
                s1_list[j] = '0' if s1_list[j] == '1' else '1'
                s1 = ''.join(s1_list)
                
                # Count the number of differences between s1 and s2
                diff = sum(1 for c1, c2 in zip(s1, s2) if c1 != c2)
                
                # If the number of differences is even, try to fix them using the second operation
                if diff % 2 == 0:
                    for k in range(n - 1):
                        if s1[k] != s2[k]:
                            s1_list = list(s1)
                            s1_list[k] = '0' if s1_list[k] == '1' else '1'
                            s1_list[k + 1] = '0' if s1_list[k + 1] == '1' else '1'
                            s1 = ''.join(s1_list)
                            cost += 1
                            diff -= 2
                            if diff == 0:
                                min_cost = min(min_cost, cost)
                                break
                
        # If it's not possible to make the strings equal, return -1
        return min_cost if min_cost != float('inf') else -1