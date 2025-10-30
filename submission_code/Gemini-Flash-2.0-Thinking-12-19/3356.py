class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = []
        for i in range(n):
            current_str = arr[i]
            shortest_substring = ""
            min_len = float('inf')
            for length in range(1, len(current_str) + 1):
                for start_index in range(len(current_str) - length + 1):
                    substring = current_str[start_index:start_index + length]
                    is_unique = True
                    for j in range(n):
                        if i != j and substring in arr[j]:
                            is_unique = False
                            break
                    if is_unique:
                        if length < min_len:
                            min_len = length
                            shortest_substring = substring
                        elif length == min_len:
                            if shortest_substring == "" or substring < shortest_substring:
                                shortest_substring = substring
            answer.append(shortest_substring)
        return answer