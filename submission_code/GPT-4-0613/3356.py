class Solution:
    def shortestSubstrings(self, arr):
        n = len(arr)
        answer = [''] * n
        for i in range(n):
            for length in range(1, len(arr[i]) + 1):
                substrings = [arr[i][j:j+length] for j in range(len(arr[i]) - length + 1)]
                substrings.sort()
                for substring in substrings:
                    if all(substring not in arr[j] for j in range(n) if j != i):
                        answer[i] = substring
                        break
                if answer[i]:
                    break
        return answer