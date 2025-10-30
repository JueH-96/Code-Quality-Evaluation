class Solution:
    def findLatestTime(self, s: str) -> str:
        
        # Helper function (can be an inner function or a private method)
        # to check if a concrete time string (e.g., "11:54")
        # can be formed from the pattern string s (e.g., "1?:?4").
        def check_match(time_str_candidate: str, pattern_str: str) -> bool:
            # The time string is always of length 5, e.g., "HH:MM"
            for i in range(5):
                # s[2] is always ':', and candidate_time[2] will also be ':'.
                # The check correctly handles this:
                # pattern_str[2] is not '?', so pattern_str[2] must equal candidate_time[2].
                # ':' == ':' is true.
                if pattern_str[i] != '?' and pattern_str[i] != time_str_candidate[i]:
                    return False
            return True

        # Iterate hours from 11 down to 0 (to find the latest valid hour first)
        for h in range(11, -1, -1):
            # Iterate minutes from 59 down to 0 (to find the latest valid minute for that hour)
            for m in range(59, -1, -1):
                # Format current h and m into "HH" and "MM" strings respectively.
                # str(num).zfill(2) ensures two digits, with a leading zero if needed.
                # e.g., str(7).zfill(2) results in "07"
                # e.g., str(11).zfill(2) results in "11"
                h_str = str(h).zfill(2)
                m_str = str(m).zfill(2)
                
                # Construct the candidate time string in "HH:MM" format
                candidate_time = f"{h_str}:{m_str}" 
                # Alternative for older Python versions: candidate_time = h_str + ":" + m_str
                
                if check_match(candidate_time, s):
                    # Since we iterate from the latest possible time downwards,
                    # the first match found is guaranteed to be the latest valid time.
                    return candidate_time
        
        # This part of the code should be unreachable based on the problem constraint:
        # "The input is generated such that there is at least one time ... that you can obtain".
        # However, to satisfy the function signature that expects a string return in all paths:
        return "" # Fallback, though not expected to be hit in practice.