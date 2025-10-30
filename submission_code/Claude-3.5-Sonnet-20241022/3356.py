class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = []
        
        def check_substring(sub, idx):
            for i in range(n):
                if i != idx and sub in arr[i]:
                    return False
            return True
            
        for i in range(n):
            s = arr[i]
            found = False
            curr_ans = ""
            
            # Try all possible lengths starting from 1
            for length in range(1, len(s) + 1):
                # Try all possible substrings of current length
                candidates = []
                for j in range(len(s) - length + 1):
                    sub = s[j:j+length]
                    if check_substring(sub, i):
                        candidates.append(sub)
                
                if candidates:
                    curr_ans = min(candidates)
                    found = True
                    break
            
            if not found:
                answer.append("")
            else:
                answer.append(curr_ans)
                
        return answer