class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = []
        for i in range(n):
            current_string = arr[i]
            shortest_unique_substring = ""
            found_unique = False
            for length in range(1, len(current_string) + 1):
                unique_substrings_of_length = []
                for start_index in range(len(current_string) - length + 1):
                    substring = current_string[start_index:start_index + length]
                    is_unique = True
                    for j in range(n):
                        if i != j and substring in arr[j]:
                            is_unique = False
                            break
                    if is_unique:
                        unique_substrings_of_length.append(substring)
                if unique_substrings_of_length:
                    shortest_unique_substring_for_length = min(unique_substrings_of_length)
                    if not shortest_unique_substring or len(shortest_unique_substring_for_length) < len(shortest_unique_substring) or (len(shortest_unique_substring_for_length) == len(shortest_unique_substring) and shortest_unique_substring_for_length < shortest_unique_substring):
                        shortest_unique_substring = shortest_unique_substring_for_length
                    found_unique = True
                    break # Since we found a unique substring of this length, any longer substring will not be shorter.
            answer.append(shortest_unique_substring)
        return answer