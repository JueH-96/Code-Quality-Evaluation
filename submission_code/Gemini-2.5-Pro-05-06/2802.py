class Solution:
    # Instance variables to store the string for i*i and the memoization table.
    # These are reset for each 'i' in the main punishmentNumber method.
    # self.s_str: str
    # self.memo: dict

    def _check(self, idx: int, target: int) -> bool:
        # Base case: if target sum becomes negative, this path is invalid.
        if target < 0:
            return False
        
        # Base case: if we've successfully parsed the entire string s_str
        if idx == len(self.s_str):
            # It's a valid partition if the target sum is exactly 0
            return target == 0
        
        # Memoization: if this state (idx, target) has been computed before, return stored result
        state = (idx, target)
        if state in self.memo:
            return self.memo[state]

        current_val_of_part = 0
        # Explore partitions: try to form the first part of s_str[idx:]
        # k is the end index (inclusive) of the current part being formed (s_str[idx...k])
        for k in range(idx, len(self.s_str)):
            # Build the integer value of this part s_str[idx...k] by appending current digit
            current_val_of_part = current_val_of_part * 10 + int(self.s_str[k])
            
            # Optimization: If the integer value of the current part (s_str[idx...k])
            # already exceeds the remaining target sum, then making this part longer
            # (by appending more digits) will only create even larger values.
            # So, we can stop trying to extend this current part.
            if current_val_of_part > target:
                break 
            
            # Recursively check if the rest of the string (s_str[k+1:])
            # can be partitioned to sum up to the new remaining target (target - current_val_of_part)
            if self._check(k + 1, target - current_val_of_part):
                self.memo[state] = True # Cache and return True if a valid partition is found
                return True
        
        # If no way to partition s_str[idx:] to sum to target, cache and return False
        self.memo[state] = False
        return False

    def punishmentNumber(self, n: int) -> int:
        total_punishment_val = 0
        for i in range(1, n + 1):
            # Set up instance variables for the current number 'i'
            self.s_str = str(i * i) # String representation of i*i
            self.memo = {}          # Reset memoization table for each 'i'
            
            # Check if 'i' satisfies the punishment property
            # Initial call: start at index 0 of s_str, target sum is 'i'
            if self._check(0, i):
                total_punishment_val += (i * i) # Add i*i to the sum
                
        return total_punishment_val