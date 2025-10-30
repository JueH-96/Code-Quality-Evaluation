class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7

        def isStepping(n):
            s = str(n)
            for i in range(len(s) - 1):
                if abs(int(s[i]) - int(s[i+1])) != 1:
                    return False
            return True

        low_num = int(low)
        high_num = int(high)
        count = 0

        
        q = deque()
        for i in range(1,10):
            q.append(i)

        while q:
            curr = q.popleft()
            if low_num <= curr <= high_num:
                count +=1
            if curr > high_num:
                continue
            last_digit = curr % 10
            if last_digit > 0:
                next_num = curr * 10 + last_digit -1
                q.append(next_num)
            if last_digit < 9:
                next_num = curr * 10 + last_digit + 1
                q.append(next_num)

        return count

from collections import deque