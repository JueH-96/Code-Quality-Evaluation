class Solution:
    def distanceTraveled(self, main: int, additional: int) -> int:
        transfers = min(main // 5, additional)
        distance = transfers * 60
        main -= 5 * transfers
        main += transfers
        return distance