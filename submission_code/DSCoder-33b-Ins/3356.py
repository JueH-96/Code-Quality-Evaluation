class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = [""] * n
        for i in range(n):
            for j in range(len(arr[i])):
                for k in range(j + 1, len(arr[i]) + 1):
                    substring = arr[i][j:k]
                    if all(substring not in arr[l] for l in range(n) if l != i):
                        if not answer[i] or len(substring) < len(answer[i]) or (len(substring) == len(answer[i]) and substring < answer[i]):
                            answer[i] = substring
        return answer