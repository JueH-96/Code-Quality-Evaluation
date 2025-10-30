class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def get_all_substrings(s):
            """Generate all possible substrings of a given string s."""
            substrings = set()
            length = len(s)
            for start in range(length):
                for end in range(start + 1, length + 1):
                    substrings.add(s[start:end])
            return substrings

        n = len(arr)
        answer = [""] * n
        
        # Generate all substrings for each string in arr
        all_substrings = [get_all_substrings(s) for s in arr]
        
        for i in range(n):
            current_substrings = all_substrings[i]
            other_substrings = set()
            
            # Collect all substrings from other strings
            for j in range(n):
                if i != j:
                    other_substrings.update(all_substrings[j])
            
            # Find the shortest and lexicographically smallest substring
            shortest_unique_substring = None
            for substring in current_substrings:
                if substring not in other_substrings:
                    if (shortest_unique_substring is None or
                        len(substring) < len(shortest_unique_substring) or
                        (len(substring) == len(shortest_unique_substring) and substring < shortest_unique_substring)):
                        shortest_unique_substring = substring
            
            # Assign the found substring or empty string if none found
            answer[i] = shortest_unique_substring if shortest_unique_substring is not None else ""
        
        return answer