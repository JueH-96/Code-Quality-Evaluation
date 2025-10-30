class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        result = []
        
        for i in range(n):
            current_string = arr[i]
            min_length = float('inf')
            best_substring = ""
            
            # Try all possible substrings of current_string
            for length in range(1, len(current_string) + 1):
                for start in range(len(current_string) - length + 1):
                    substring = current_string[start:start + length]
                    
                    # Check if this substring appears in any other string
                    found_in_other = False
                    for j in range(n):
                        if i != j and substring in arr[j]:
                            found_in_other = True
                            break
                    
                    # If not found in any other string
                    if not found_in_other:
                        # Update if this is shorter, or same length but lexicographically smaller
                        if length < min_length or (length == min_length and substring < best_substring):
                            min_length = length
                            best_substring = substring
            
            result.append(best_substring)
        
        return result