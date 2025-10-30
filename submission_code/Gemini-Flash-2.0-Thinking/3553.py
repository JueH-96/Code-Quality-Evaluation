class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        col1 = coordinate1[0]
        row1 = int(coordinate1[1])

        col2 = coordinate2[0]
        row2 = int(coordinate2[1])

        col_index1 = ord(col1) - ord('a')
        row_index1 = row1 - 1

        col_index2 = ord(col2) - ord('a')
        row_index2 = row2 - 1

        sum1 = col_index1 + row_index1
        sum2 = col_index2 + row_index2

        return sum1 % 2 == sum2 % 2