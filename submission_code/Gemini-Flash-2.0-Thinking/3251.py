class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_diagonal_squared = -1
        max_area = -1

        for dimension in dimensions:
            length = dimension[0]
            width = dimension[1]

            current_diagonal_squared = length * length + width * width
            current_area = length * width

            if current_diagonal_squared > max_diagonal_squared:
                max_diagonal_squared = current_diagonal_squared
                max_area = current_area
            elif current_diagonal_squared == max_diagonal_squared:
                if current_area > max_area:
                    max_area = current_area

        return max_area