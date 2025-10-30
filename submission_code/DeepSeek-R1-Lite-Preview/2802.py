class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_split(s, target, index, current_sum, memo):
            if (index, current_sum) in memo:
                return memo[(index, current_sum)]
            if index == len(s):
                return current_sum == target
            for i in range(index, len(s)):
                substring = s[index:i+1]
                number = int(substring)
                if can_split(s, target, i+1, current_sum + number, memo):
                    memo[(index, current_sum)] = True
                    return True
            memo[(index, current_sum)] = False
            return False
        
        punishment_sum = 0
        for i in range(1, n+1):
            square = i * i
            s = str(square)
            memo = {}
            if can_split(s, i, 0, 0, memo):
                punishment_sum += square
        return punishment_sum