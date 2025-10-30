class Solution:
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        col1_char = coordinate1[0]
        row1_char = coordinate1[1]
        col2_char = coordinate2[0]
        row2_char = coordinate2[1]

        col1_index = ord(col1_char) - ord('a')
        row1_index = int(row1_char)
        col2_index = ord(col2_char) - ord('a')
        row2_index = int(row2_char)

        sum1 = col1_index + row1_index
        sum2 = col2_index + row2_index

        if sum1 % 2 == sum2 % 2:
            return True
        else:
            return False