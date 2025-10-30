class Solution:
    def isBalanced(self, num: str) -> bool:
        # Sum digits at even indices: 0,2,4,...
        even_sum = sum(int(num[i]) for i in range(0, len(num), 2))
        # Sum digits at odd indices: 1,3,5,...
        odd_sum = sum(int(num[i]) for i in range(1, len(num), 2))
        return even_sum == odd_sum