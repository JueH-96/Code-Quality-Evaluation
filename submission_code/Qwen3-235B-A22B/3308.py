class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Initialize times for each character a-z
        times = [0] * 26
        # Initialize max_prev for each character to track the maximum time of lower codes so far
        max_prev = [0] * 26
        removal_times = []
        max_k = 0
        
        for c_char in s:
            c = ord(c_char) - ord('a')
            # Compute the current_max_lower by checking all characters less than c
            current_max_lower = 0
            for i in range(c):
                if max_prev[i] > current_max_lower:
                    current_max_lower = max_prev[i]
            # Calculate current time
            current_time = max(times[c] + 1, current_max_lower)
            removal_times.append(current_time)
            # Update max_k
            if current_time > max_k:
                max_k = current_time
            # Update times and max_prev for character c
            times[c] = current_time
            if current_time > max_prev[c]:
                max_prev[c] = current_time
        
        # Collect characters with removal time equal to max_k
        result = []
        for i in range(len(s)):
            if removal_times[i] == max_k:
                result.append(s[i])
        
        return ''.join(result)