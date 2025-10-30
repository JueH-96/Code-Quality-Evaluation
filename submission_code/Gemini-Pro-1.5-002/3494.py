class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort()
        verticalCut.sort()
        
        horiz = [0]
        for cut in horizontalCut:
            horiz.append(cut)
        horiz.append(0)

        vert = [0]
        for cut in verticalCut:
            vert.append(cut)
        vert.append(0)
        
        for i in range(1, len(horiz)):
            horiz[i] += horiz[i-1]
            
        for i in range(1, len(vert)):
            vert[i] += vert[i-1]
            
        
        ans = 0
        
        h = len(horiz) - 1
        v = len(vert) - 1
        
        while h > 0 or v > 0:
            if (h > 0 and (v == 0 or horiz[h] - horiz[h-1] > vert[v] - vert[v-1])):
                cost = horiz[h] - horiz[h-1]
                ans += cost * v
                h -= 1
            elif v > 0:
                cost = vert[v] - vert[v-1]
                ans += cost * h
                v -= 1
        
        return ans