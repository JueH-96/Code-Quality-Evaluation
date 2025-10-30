class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        
        max_sum = float('-inf')
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    if m >= 3:
                        current_sum = board[0][i] + board[1][j] + board[2][k]
                        max_sum = max(max_sum, current_sum)
                    
                    
                    if m > 3:
                        import itertools
                        rows = list(range(m))
                        cols = [i,j,k]
                        
                        if len(set(cols)) == 3:
                            
                            possible_sums = []
                            
                            for row_indices in itertools.permutations(rows, 3):
                                current_sum = 0
                                valid = True
                                
                                current_sum += board[row_indices[0]][i]
                                current_sum += board[row_indices[1]][j]
                                current_sum += board[row_indices[2]][k]
                                
                                possible_sums.append(current_sum)
                            
                            if possible_sums:
                                max_sum = max(max_sum, max(possible_sums))
        
        return max_sum