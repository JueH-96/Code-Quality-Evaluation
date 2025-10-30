class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def get_all_substrings(s):
            substrings = set()
            n = len(s)
            for length in range(1, n+1):
                for i in range(n - length + 1):
                    substrings.add(s[i:i+length])
            return substrings
        
        # Precompute all substrings for each string in arr
        all_substrings = []
        for s in arr:
            all_substrings.append(get_all_substrings(s))
        
        answer = []
        for i in range(len(arr)):
            s = arr[i]
            n = len(s)
            found = False
            # Try all possible lengths starting from 1
            for length in range(1, n+1):
                candidates = set()
                for j in range(n - length + 1):
                    substring = s[j:j+length]
                    # Check if this substring is not in any other string's substrings
                    unique = True
                    for k in range(len(arr)):
                        if k != i and substring in all_substrings[k]:
                            unique = False
                            break
                    if unique:
                        candidates.add(substring)
                if candidates:
                    # Choose the lex smallest
                    chosen = min(candidates)
                    answer.append(chosen)
                    found = True
                    break
            if not found:
                answer.append("")
        return answer