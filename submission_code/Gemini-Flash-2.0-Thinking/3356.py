class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = [""] * n

        for i in range(n):
            current_string = arr[i]
            shortest_unique = None

            for length in range(1, len(current_string) + 1):
                found_unique_for_length = False
                substrings = set()
                for k in range(len(current_string) - length + 1):
                    substring = current_string[k : k + length]
                    is_unique = True
                    for j in range(n):
                        if i != j and substring in arr[j]:
                            is_unique = False
                            break
                    if is_unique:
                        if shortest_unique is None:
                            shortest_unique = substring
                        elif len(substring) < len(shortest_unique):
                            shortest_unique = substring
                        elif len(substring) == len(shortest_unique) and substring < shortest_unique:
                            shortest_unique = substring
                        found_unique_for_length = True

                if found_unique_for_length and shortest_unique is not None and len(shortest_unique) == length:
                    break # Optimization: if we found a unique substring of this length, no need to check longer ones

            if shortest_unique is not None:
                answer[i] = shortest_unique

        return answer