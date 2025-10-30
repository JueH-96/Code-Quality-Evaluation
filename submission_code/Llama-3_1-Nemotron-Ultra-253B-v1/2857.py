class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        distance = 0
        main = mainTank
        additional = additionalTank
        while main > 0:
            if main >= 5:
                distance += 50
                main -= 5
                if additional >= 1:
                    main += 1
                    additional -= 1
            else:
                distance += main * 10
                main = 0
        return distance