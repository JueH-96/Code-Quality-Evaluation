class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        count = 0
        for x in range(1, n + 1):
            for y in range(1, m + 1):
                total_flowers = x + y
                if total_flowers % 2 != 0:
                    if x > y:
                        count +=1
                    
                elif total_flowers %2 == 0 and x !=y:
                    if x > y:
                        count +=1

        return count