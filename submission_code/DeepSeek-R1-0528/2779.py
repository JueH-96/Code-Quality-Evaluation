class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        arr = [0] * n
        total = 0
        ans = []
        
        for idx, color in queries:
            old = arr[idx]
            delta = 0
            
            if idx > 0:
                if old != 0 and arr[idx-1] == old:
                    delta -= 1
                if color != 0 and arr[idx-1] == color:
                    delta += 1
            
            if idx < n - 1:
                if old != 0 and arr[idx+1] == old:
                    delta -= 1
                if color != 0 and arr[idx+1] == color:
                    delta += 1
            
            arr[idx] = color
            total += delta
            ans.append(total)
        
        return ans