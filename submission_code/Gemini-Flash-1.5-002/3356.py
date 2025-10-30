class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        ans = [""] * n
        for i in range(n):
            s = arr[i]
            shortest = ""
            for j in range(len(s)):
                for k in range(j, len(s)):
                    sub = s[j:k+1]
                    found = False
                    for l in range(n):
                        if l != i and sub in arr[l]:
                            found = True
                            break
                    if not found:
                        if shortest == "" or len(sub) < len(shortest) or (len(sub) == len(shortest) and sub < shortest):
                            shortest = sub
            ans[i] = shortest
        return ans