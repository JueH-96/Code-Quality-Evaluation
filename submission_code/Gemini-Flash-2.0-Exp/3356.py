class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        ans = [""] * n
        for i in range(n):
            shortest_len = float('inf')
            shortest_str = ""
            for length in range(1, len(arr[i]) + 1):
                for j in range(len(arr[i]) - length + 1):
                    sub = arr[i][j:j+length]
                    found = False
                    for k in range(n):
                        if i != k and sub in arr[k]:
                            found = True
                            break
                    if not found:
                        if length < shortest_len:
                            shortest_len = length
                            shortest_str = sub
                        elif length == shortest_len and sub < shortest_str:
                            shortest_str = sub
            ans[i] = shortest_str
        return ans