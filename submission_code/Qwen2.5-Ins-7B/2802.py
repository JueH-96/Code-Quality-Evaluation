class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_valid(i):
            square = str(i * i)
            for size in range(1, len(square) + 1):
                for start in range(len(square) - size + 1):
                    parts = [int(square[start:start+size])]
                    if sum(parts) == i:
                        return True
                    for split in range(1, size):
                        parts = [int(square[start:start+split]), int(square[start+split:start+size])]
                        if sum(parts) == i:
                            return True
            return False
        
        total = 0
        for i in range(1, n + 1):
            if is_valid(i):
                total += i * i
        return total