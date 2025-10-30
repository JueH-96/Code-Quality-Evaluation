class Solution:
    def maximumLength(self, s: str) -> int:
        groups = []
        n = len(s)
        current_char = s[0]
        count = 1
        for i in range(1, n):
            if s[i] == current_char:
                count += 1
            else:
                groups.append((current_char, count))
                current_char = s[i]
                count = 1
        groups.append((current_char, count))
        
        run_dict = {}
        for char, cnt in groups:
            if char not in run_dict:
                run_dict[char] = []
            run_dict[char].append(cnt)
        
        if not groups:
            return -1
        
        max_run = max(cnt for _, cnt in groups)
        
        ans = -1
        for k in range(max_run, 0, -1):
            found = False
            for char in run_dict:
                total_occurrences = 0
                for run in run_dict[char]:
                    if run >= k:
                        total_occurrences += (run - k + 1)
                if total_occurrences >= 3:
                    ans = k
                    found = True
                    break
            if found:
                break
                
        return ans