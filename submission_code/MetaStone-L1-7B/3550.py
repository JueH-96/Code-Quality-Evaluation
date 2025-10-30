class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        max_sum = -float('inf')
        m = len(board)
        if m == 0:
            return 0
        n = len(board[0])
        for i in range(m):
            for j in range(n):
                # Find max1 in the collection
                max1 = -float('inf')
                max1_x, max1_y = -1, -1
                for x in range(m):
                    if x == i:
                        continue
                    for y in range(n):
                        if y == j:
                            continue
                        val = board[x][y]
                        if val > max1:
                            max1 = val
                            max1_x, max1_y = x, y
                # Find max2 in the collection excluding max1's row and column
                max2 = -float('inf')
                max2_x, max2_y = -1, -1
                for x in range(m):
                    if x == i or x == max1_x:
                        continue
                    for y in range(n):
                        if y == j or y == max1_y:
                            continue
                        val = board[x][y]
                        if val > max2:
                            max2 = val
                            max2_x, max2_y = x, y
                # Calculate current sum
                current_sum = board[i][j] + max1 + max2
                if current_sum > max_sum:
                    max_sum = current_sum
        return max_sum