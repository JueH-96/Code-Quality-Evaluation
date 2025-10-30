class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        total_count = 0
        s_int = [int(c) for c in s]

        for j in range(n):
            last_digit_val = s_int[j]

            # If the last digit is '0', it cannot be used as a divisor.
            if last_digit_val == 0:
                continue

            d = last_digit_val
            
            # current_num_val_mod_d stores (value of s[i...j]) % d
            # It's accumulated from right to left (from s[j] to s[i]).
            current_num_val_mod_d = 0 
            
            # power_of_10 stores 10^(position_from_right) % d.
            # For s[j], position_from_right is 0 (10^0).
            # For s[j-1], position_from_right is 1 (10^1).
            # ...
            # For s[i], position_from_right is j-i (10^(j-i)).
            power_of_10 = 1 # Represents 10^0 initially

            # Iterate from the current ending digit (j) back to the start of the string (0)
            # 'i' represents the starting index of the substring s[i...j]
            for i in range(j, -1, -1):
                digit_at_i = s_int[i]
                
                # Update current_num_val_mod_d:
                # new_value = digit_at_i * 10^(j-i) + value_of_s[i+1...j]
                current_num_val_mod_d = (digit_at_i * power_of_10 + current_num_val_mod_d) % d
                
                # If the current substring s[i...j] is divisible by d, increment count
                if current_num_val_mod_d == 0:
                    total_count += 1
                
                # Optimization for d=2, 4, 5, 8:
                # For these divisors, 10^k % d eventually becomes 0.
                # Once it becomes 0, any digit multiplied by this power will be 0 mod d.
                # This means further digits to the left (smaller 'i') will not change the remainder
                # if the remainder is already 0.
                if d in {2, 4, 5, 8}:
                    # Calculate power_of_10 for the next step (i.e., for s[i-1...j])
                    next_power_of_10 = (power_of_10 * 10) % d
                    if next_power_of_10 == 0:
                        # If the next power of 10 will be 0, it means 10^(j-(i-1)) % d == 0.
                        # This also implies 10^k % d == 0 for all k > (j-i).
                        # If the current substring s[i...j] is divisible by d (current_num_val_mod_d == 0),
                        # then all longer substrings starting before 'i' (s[i'...j] for i' < i)
                        # will also be divisible by d.
                        if current_num_val_mod_d == 0:
                            # Add the count for all remaining starting positions (from 0 to i-1)
                            total_count += i 
                        break # No need to check further to the left
                
                # Update power_of_10 for the next iteration (when 'i' decreases by 1)
                power_of_10 = (power_of_10 * 10) % d
                
        return total_count