class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def is_substring(sub, s):
            return sub in s

        n = len(arr)
        answer = []

        for i in range(n):
            current_string = arr[i]
            min_substring = ""

            for length in range(1, len(current_string) + 1):
                for start in range(len(current_string) - length + 1):
                    substring = current_string[start:start + length]
                    is_unique = True
                    for j in range(n):
                        if j != i and is_substring(substring, arr[j]):
                            is_unique = False
                            break
                    if is_unique:
                        if min_substring == "" or len(substring) < len(min_substring) or (len(substring) == len(min_substring) and substring < min_substring):
                            min_substring = substring

            answer.append(min_substring)

        return answer