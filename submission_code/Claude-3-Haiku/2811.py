class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Initialize the minimum sum to 0
        min_sum = 0
        
        # Construct the k-avoiding array
        for i in range(1, n+1):
            # Find the smallest positive integer that is not a part of the current sum
            # and does not sum to k with any element in the array
            j = 1
            while True:
                if all(j != min_sum + x for x in range(1, i)):
                    if all(j != k - x for x in range(1, i)):
                        break
                j += 1
            min_sum += j
        
        return min_sum