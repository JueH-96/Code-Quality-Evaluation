class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        zeros = num.count('0')
        min_ops = n
        
        for i in range(n-1, -1, -1):
            if num[i] == '0':
                min_ops = min(min_ops, n - zeros)
                for j in range(i-1, -1, -1):
                    if num[j] in '05':
                        min_ops = min(min_ops, n - (i - j + 1 + zeros - (num[j:i+1].count('0') if '0' in num[j:i+1] else 0)))
                        break
            elif num[i] == '5':
                min_ops = min(min_ops, n - zeros)
                for j in range(i-1, -1, -1):
                    if num[j] in '27':
                        min_ops = min(min_ops, n - (i - j + 1 + zeros - (num[j:i+1].count('0') if '0' in num[j:i+1] else 0)))
                        break
        
        return min_ops