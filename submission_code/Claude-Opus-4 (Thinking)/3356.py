class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        result = []
        
        # For each string in arr
        for i in range(n):
            current_string = arr[i]
            
            # Collect all substrings from other strings
            other_substrings = set()
            for j in range(n):
                if i != j:
                    other_string = arr[j]
                    # Generate all substrings of other_string
                    for start in range(len(other_string)):
                        for end in range(start + 1, len(other_string) + 1):
                            other_substrings.add(other_string[start:end])
            
            # Find the shortest substring of current_string that's not in other_substrings
            best_substring = None
            
            # Check substrings by length (shortest first)
            for length in range(1, len(current_string) + 1):
                candidates = []
                for start in range(len(current_string) - length + 1):
                    substring = current_string[start:start + length]
                    if substring not in other_substrings:
                        candidates.append(substring)
                
                if candidates:
                    # Found substrings of this length that don't appear in other strings
                    # Choose the lexicographically smallest one
                    best_substring = min(candidates)
                    break
            
            if best_substring is None:
                result.append("")
            else:
                result.append(best_substring)
        
        return result