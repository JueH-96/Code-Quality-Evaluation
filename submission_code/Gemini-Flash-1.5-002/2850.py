class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        ans = 0
        for i in range(min(x,y)+1):
            cur_x = x -i
            cur_y = y -i
            cur_z = z
            
            temp_ans = 2*i + 2*min(cur_x,cur_y) + cur_z
            
            if cur_x > cur_y:
                temp_ans += 2*(cur_x - cur_y -1) if cur_x - cur_y >1 else 0
                
            elif cur_y > cur_x:
                temp_ans += 2*(cur_y - cur_x -1) if cur_y - cur_x > 1 else 0
            
            
            
            
            if cur_x > 0 and cur_y > 0:
                temp_ans = max(temp_ans, 2*min(cur_x,cur_y) + cur_z + 2*(max(cur_x,cur_y) - min(cur_x,cur_y) -1) if max(cur_x,cur_y) - min(cur_x,cur_y) > 1 else 2*min(cur_x,cur_y) + cur_z)
            
            
            ans = max(ans,temp_ans)
            
        
        return ans