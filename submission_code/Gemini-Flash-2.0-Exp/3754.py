class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        max_dist = 0
        x, y = 0, 0
        
        for i in range(n):
            temp_x, temp_y = x, y
            
            best_dist = 0
            
            for j in range(k + 1):
                curr_x, curr_y = x, y
                curr_k = j
                
                for l in range(i, n):
                    curr_s = s[l]
                    
                    if curr_s == 'N':
                        curr_y += 1
                    elif curr_s == 'S':
                        curr_y -= 1
                    elif curr_s == 'E':
                        curr_x += 1
                    else:
                        curr_x -= 1
                        
                best_dist = max(best_dist, abs(curr_x) + abs(curr_y))
            
            if s[i] == 'N':
                y += 1
            elif s[i] == 'S':
                y -= 1
            elif s[i] == 'E':
                x += 1
            else:
                x -= 1
            
            max_dist = max(max_dist, abs(x) + abs(y))
            
        
        
        dp = {}
        
        def solve(index, changes, x, y):
            if index == len(s):
                return abs(x) + abs(y)
            
            if (index, changes, x, y) in dp:
                return dp[(index, changes, x, y)]
            
            ans = 0
            
            # Option 1: Don't change the character
            if s[index] == 'N':
                ans = max(ans, solve(index + 1, changes, x, y + 1))
            elif s[index] == 'S':
                ans = max(ans, solve(index + 1, changes, x, y - 1))
            elif s[index] == 'E':
                ans = max(ans, solve(index + 1, changes, x + 1, y))
            else:
                ans = max(ans, solve(index + 1, changes, x - 1, y))
            
            # Option 2: Change the character
            if changes > 0:
                ans = max(ans, solve(index + 1, changes - 1, x + 1, y)) # Change to E
                ans = max(ans, solve(index + 1, changes - 1, x - 1, y)) # Change to W
                ans = max(ans, solve(index + 1, changes - 1, x, y + 1)) # Change to N
                ans = max(ans, solve(index + 1, changes - 1, x, y - 1)) # Change to S
            
            dp[(index, changes, x, y)] = ans
            return ans
        
        return solve(0, k, 0, 0)