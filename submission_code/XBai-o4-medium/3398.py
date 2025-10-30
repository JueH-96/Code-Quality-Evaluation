from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        for i in range(2):
            for j in range(2):
                cnt_b = 0
                cnt_w = 0
                for x in range(2):
                    for y in range(2):
                        if grid[i + x][j + y] == 'B':
                            cnt_b += 1
                        else:
                            cnt_w += 1
                if cnt_b >= 3 or cnt_w >= 3:
                    return True
        return False