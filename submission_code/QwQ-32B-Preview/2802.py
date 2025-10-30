class Solution:
    def punishmentNumber(self, n: int) -> int:
        total_sum = 0
        for i in range(1, n + 1):
            sq = i * i
            sq_str = str(sq)
            target = i
            memo = {}
            if self.can_partition(sq_str, target, 0, 0, memo):
                total_sum += sq
        return total_sum

    def can_partition(self, sq_str, target, start, current_sum, memo):
        if start == len(sq_str):
            return current_sum == target
        if (start, current_sum) in memo:
            return memo[(start, current_sum)]
        
        for end in range(start, len(sq_str)):
            part = int(sq_str[start:end+1])
            if self.can_partition(sq_str, target, end + 1, current_sum + part, memo):
                memo[(start, current_sum)] = True
                return True
        memo[(start, current_sum)] = False
        return False