class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        for i in range(2):
            for j in range(2):
                b_count = 0
                w_count = 0
                for x in range(i, i+2):
                    for y in range(j, j+2):
                        if grid[x][y] == 'B':
                            b_count +=1
                        else:
                            w_count +=1
                if b_count <=1 or w_count <=1:
                    return True
        return False