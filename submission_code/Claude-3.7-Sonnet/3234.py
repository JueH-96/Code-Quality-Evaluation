class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        
        for i, [a, b, c, m] in enumerate(variables):
            # Step 1: Calculate (a_i^b_i % 10)
            step1 = pow(a, b, 10)
            
            # Step 2: Calculate ((a_i^b_i % 10)^c_i) % m_i
            step2 = pow(step1, c, m)
            
            # Check if the result equals the target
            if step2 == target:
                good_indices.append(i)
                
        return good_indices