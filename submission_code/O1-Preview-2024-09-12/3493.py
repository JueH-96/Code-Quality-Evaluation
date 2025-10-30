class Solution:
    def maxOperations(self, s: str) -> int:
        zeros = 0
        inversions = 0
        adjacent_ones = 0
        n = len(s)
        # Count inversions (number of zeros after ones)
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                zeros += 1
            else:  # s[i] == '1'
                inversions += zeros
        # Count adjacent ones
        for i in range(n - 1):
            if s[i] == '1' and s[i + 1] == '1':
                adjacent_ones += 1
        # Calculate the maximum number of operations
        answer = inversions - adjacent_ones
        if answer < 0:
            answer = 0
        return answer