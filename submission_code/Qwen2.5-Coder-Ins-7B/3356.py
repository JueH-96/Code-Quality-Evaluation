class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = [""] * n
        
        for i in range(n):
            substrings = set()
            for j in range(len(arr[i])):
                for k in range(j + 1, len(arr[i]) + 1):
                    substrings.add(arr[i][j:k])
            
            for j in range(len(arr[i])):
                for k in range(j + 1, len(arr[i]) + 1):
                    substring = arr[i][j:k]
                    if all(substring not in arr[m] for m in range(n) if m != i):
                        if not answer[i] or len(substring) < len(answer[i]) or (len(substring) == len(answer[i]) and substring < answer[i]):
                            answer[i] = substring
        
        return answer