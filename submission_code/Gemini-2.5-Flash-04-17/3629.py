class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Initialize counts for each character 'a' through 'z'
        # counts[0] is 'a', counts[25] is 'z'
        counts = [0] * 26
        for char in s:
            counts[ord(char) - ord('a')] += 1
            
        # Perform t transformations
        for _ in range(t):
            new_counts = [0] * 26
            
            # Recurrence relation for counts based on where characters *come from*:
            # If old string has N_z 'z's, the new string gets N_z 'a's and N_z 'b's.
            # If old string has N_a 'a's, the new string gets N_a 'b's.
            # If old string has N_b 'b's, the new string gets N_b 'c's.
            # ...
            # If old string has N_y 'y's, the new string gets N_y 'z's.

            # 'a' (index 0) in new_counts receives from 'z' (index 25) in old_counts
            new_counts[0] = counts[25] % MOD
            
            # 'b' (index 1) in new_counts receives from 'a' (index 0) and 'z' (index 25) in old_counts
            new_counts[1] = (counts[0] + counts[25]) % MOD
            
            # 'c' through 'z' (indices 2 through 25) in new_counts receive from the previous character in old_counts
            # new_counts[i] receives from counts[i-1] for i = 2 to 25
            for i in range(2, 26):
                 new_counts[i] = counts[i-1] % MOD

            # Update counts for the next iteration
            counts = new_counts

        # Calculate the total length (sum of all counts) modulo MOD
        # Python's sum handles large integers, so modulo at the end is fine.
        total_length = sum(counts) % MOD
        
        return total_length