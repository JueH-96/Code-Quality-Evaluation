class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        row3max = [sum(sorted(row, reverse=True)[:3]) for row in board]
        rowmax = [max(row) for row in board]
        colmax = [max(col) for col in zip(*board)]
        midmax = []
        temp_row = []
        
        for j in range(len(board[0])):
            for i in reversed(range(len(board))):
                temp_row.append(board[i][j])
                if len(temp_row) == 2 and any(temp_row[0] == x for x in temp_row):
                    temp_row.pop(0)
                    break
            midmax.append(max(temp_row, default=0))
            temp_row.clear()
        
        best_max = max(row3max[0] - max(row3max[0], key=abs), row3max[1] - max(row3max[1], key=abs),
                        row3max[2] - max(row3max[2], key=abs))
        r1c2, r2c1 = 0, 0
        # Copilot-generated loops that might not fulfill the requirements.
        
        for i in range(len(board)):
            for j in range(1, len(board[0])):
                if board[i][j] == rowmax[i] and midmax[j] >= (row3max[i] - rowmax[i]):
                    r1c2 = max(r1c2, midmax[j] + rowmax[i] + row3max[(i + 1) % 3])
                    r2c1 = max(r2c1, midmax[j] + rowmax[i] + row3max[(i + 2) % 3])
        
        return max(best_max, r1c2, r2c1, colmax[0] + colmax[1] + row3max[2],
                   colmax[0] + colmax[2] + row3max[1],
                   colmax[1] + colmax[2] + row3max[0])