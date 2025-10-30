class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        
        # If num is "0", no operations needed
        if num == "0":
            return 0
            
        # Try to find last two digits that make number divisible by 25
        # Valid endings are: 00, 25, 50, 75
        
        # Initialize min operations to deleting all digits except one zero if exists
        # or all digits if no zero exists
        min_ops = n - (1 if '0' in num else 0)
        
        # Check for "00"
        last_zero = -1
        second_last_zero = -1
        for i in range(n-1, -1, -1):
            if num[i] == '0':
                if last_zero == -1:
                    last_zero = i
                elif second_last_zero == -1:
                    second_last_zero = i
                    min_ops = min(min_ops, n - second_last_zero - 2)
                    
        # Check for "25"
        last_five = -1
        for i in range(n-1, -1, -1):
            if num[i] == '5' and last_five == -1:
                last_five = i
            elif num[i] == '2' and last_five != -1:
                min_ops = min(min_ops, n - i - 2)
                
        # Check for "50"
        if last_five != -1 and last_zero != -1:
            for i in range(n-1, -1, -1):
                if num[i] == '5' and i < last_zero:
                    min_ops = min(min_ops, n - i - 2)
                    
        # Check for "75"
        if last_five != -1:
            for i in range(n-1, -1, -1):
                if num[i] == '7' and i < last_five:
                    min_ops = min(min_ops, n - i - 2)
                    
        return min_ops