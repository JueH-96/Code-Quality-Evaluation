class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def bfs(low, high):
            from collections import deque
            
            low = int(low)
            high = int(high)
            count = 0
            queue = deque([i for i in range(1, 10)])
            
            if low == 0:
                count += 1
            
            while queue:
                num = queue.popleft()
                
                if low <= num <= high:
                    count += 1
                    count %= MOD
                
                last_digit = num % 10
                if num <= high:
                    if last_digit > 0:
                        next_num = num * 10 + (last_digit - 1)
                        if next_num <= high:
                            queue.append(next_num)
                    if last_digit < 9:
                        next_num = num * 10 + (last_digit + 1)
                        if next_num <= high:
                            queue.append(next_num)
            
            return count
        
        return bfs(low, high)