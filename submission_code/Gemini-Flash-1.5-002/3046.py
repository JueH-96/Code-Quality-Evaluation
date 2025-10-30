class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        ans = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                s = num[i] + num[j]
                if s == "00" or s == "25" or s == "50" or s == "75":
                    cur = 0
                    temp = ""
                    for k in range(n):
                        if k != i and k != j:
                            temp += num[k]
                    
                    if len(temp) > 0:
                        
                        first_digit_index = 0
                        while first_digit_index < len(temp) and temp[first_digit_index] == '0':
                            first_digit_index +=1
                        
                        if first_digit_index == len(temp):
                            cur = n -2
                        else:
                            cur = n - 2
                    else:
                        cur = n -2

                    ans = min(ans, cur)
        
        if ans == float('inf'):
            
            ans = n -1 if len(num) > 1 else 1 if int(num) % 25 == 0 else 1
            
        
        return ans