class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = float('inf')
        
        # Try deleting each digit one by one
        for i in range(n):
            new_num = num[:i] + num[i+1:]
            if int(new_num) % 25 == 0:
                min_ops = min(min_ops, 1)
                return min_ops
        
        # Try deleting two digits
        for i in range(n):
            for j in range(i+1, n):
                new_num = num[:i] + num[i+1:j] + num[j+1:]
                if int(new_num) % 25 == 0:
                    min_ops = min(min_ops, 2)
                    return min_ops
        
        # Try deleting three digits
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    new_num = num[:i] + num[i+1:j] + num[j+1:k] + num[k+1:]
                    if int(new_num) % 25 == 0:
                        min_ops = min(min_ops, 3)
                        return min_ops
        
        # If no special number can be formed, return the length of the string
        return n