class Solution:
    def punishmentNumber(self, n: int) -> int:
        punishment_sum = 0
        for i in range(1, n+1):
            square = i * i
            square_str = str(square)
            for j in range(1, len(square_str)):
                for k in range(j+1, len(square_str)+1):
                    if int(square_str[:j]) + int(square_str[j:k]) == i:
                        punishment_sum += square
                        break
            else:
                if int(square_str) == i:
                    punishment_sum += square
        return punishment_sum