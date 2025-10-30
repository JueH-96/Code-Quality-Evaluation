from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = [""] * n

        for i in range(n):
            shortest_len = float('inf')
            best_substring = ""
            for length in range(1, len(arr[i]) + 1):
                for start in range(len(arr[i]) - length + 1):
                    substring = arr[i][start:start + length]
                    is_unique = True
                    for j in range(n):
                        if i != j and substring in arr[j]:
                            is_unique = False
                            break
                    if is_unique:
                        if length < shortest_len:
                            shortest_len = length
                            best_substring = substring
                        elif length == shortest_len and substring < best_substring:
                            best_substring = substring
            answer[i] = best_substring

        return answer