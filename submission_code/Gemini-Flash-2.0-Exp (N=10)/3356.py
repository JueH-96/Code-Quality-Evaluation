class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = [""] * n

        for i in range(n):
            shortest_len = float('inf')
            shortest_sub = ""
            for length in range(1, len(arr[i]) + 1):
                for start in range(len(arr[i]) - length + 1):
                    sub = arr[i][start:start + length]
                    found = False
                    for j in range(n):
                        if i != j and sub in arr[j]:
                            found = True
                            break
                    if not found:
                        if length < shortest_len:
                            shortest_len = length
                            shortest_sub = sub
                        elif length == shortest_len and sub < shortest_sub:
                            shortest_sub = sub
            answer[i] = shortest_sub
        return answer