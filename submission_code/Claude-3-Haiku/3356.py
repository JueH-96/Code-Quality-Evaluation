class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = []
        
        for i in range(n):
            min_len = float('inf')
            min_substr = ''
            for j in range(1, len(arr[i]) + 1):
                substr = arr[i][:j]
                if all(substr not in arr[k] for k in range(n) if k != i):
                    if len(substr) < min_len:
                        min_len = len(substr)
                        min_substr = substr
                    elif len(substr) == min_len and substr < min_substr:
                        min_substr = substr
            answer.append(min_substr)
        
        return answer