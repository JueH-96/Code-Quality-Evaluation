import math
from collections import Counter

class Solution:
    """
    Finds the minimum possible length of a string t such that s is a concatenation of anagrams of t.
    The approach involves finding divisors of the string length n, applying a necessary condition based on character counts and LCM/GCD,
    and then verifying the sufficient condition by checking if segments are anagrams.
    """
    
    @staticmethod
    def _lcm(a: int, b: int) -> int:
        """
        Calculates the least common multiple (LCM) of two non-negative integers a and b.
        Uses the formula LCM(a, b) = |a * b| / GCD(a, b).
        Handles edge cases involving zero.
        """
        # The LCM of 0 and any number is 0.
        if a == 0 or b == 0:
            return 0
        # Calculate LCM using the formula involving GCD.
        # math.gcd handles various cases, including gcd(x, 0) = |x|.
        # The initial check for a=0 or b=0 prevents potential division by zero if math.gcd(0, 0) = 0.
        return abs(a * b) // math.gcd(a, b)

    @staticmethod
    def _check_k(s: str, n: int, k: int) -> bool:
        """
        Checks if the string s of length n can be segmented into n/k substrings of length k,
        such that all substrings are anagrams of the first substring s[0:k].
        
        Args:
            s: The input string.
            n: The length of the input string s.
            k: The potential length of the base anagram string t.

        Returns:
            True if s is composed of anagrams of length k, False otherwise.
        """
        # Calculate the frequency map (character counts) for the first segment s[0:k].
        # This serves as the reference count map.
        target_count = Counter(s[0:k])
        
        # Calculate the total number of segments the string s should be divided into.
        num_substrings = n // k
        
        # Iterate through the remaining segments, starting from the second segment (index 1).
        for i in range(1, num_substrings):
            # Define the start and end indices for the current segment.
            start_index = i * k
            end_index = (i + 1) * k
            
            # Calculate the frequency map for the current segment s[start_index:end_index].
            # String slicing s[start:end] takes O(k) time.
            # Creating a Counter from a string of length k takes O(k) time.
            current_count = Counter(s[start_index:end_index])
            
            # Compare the frequency map of the current segment with the target frequency map.
            # Counter comparison takes O(alphabet_size) time, which is constant (O(1)) for lowercase English letters.
            if current_count != target_count:
                # If the counts differ, the segments are not anagrams. Return False immediately.
                return False
                
        # If the loop completes without finding any non-anagram segment, all segments are anagrams.
        return True

    def minAnagramLength(self, s: str) -> int:
        """
        Calculates the minimum possible length of a string t such that s is a concatenation of anagrams of t.
        
        Args:
            s: The input string.

        Returns:
            The minimum possible length of string t.
        """
        n = len(s)
        
        # Calculate the frequency map for all characters in the entire string s.
        total_count_map = Counter(s)
        
        # Calculate a necessary condition for the length k based on character counts and number theory.
        # For s to be a concatenation of anagrams of t (length k), the total count of each character `c`
        # in `s` must be a multiple of `n/k`. This implies `k * total_count[c]` is divisible by `n`.
        # This leads to the condition that k must be a multiple of `n / gcd(n, total_count[c])` for all characters c.
        # Therefore, k must be a multiple of the LCM of these terms over all characters.
        required_lcm = 1
        for char_count in total_count_map.values():
             # Find the greatest common divisor (GCD) between string length n and the character count.
            common_divisor = math.gcd(n, char_count)
            # Calculate the term n / gcd(n, char_count).
            term = n // common_divisor
            # Update the required LCM by taking the LCM of the current required_lcm and the term.
            # Any valid k must be a multiple of this final required_lcm.
            required_lcm = Solution._lcm(required_lcm, term) 
        
        # Find all divisors of n. These are the potential candidates for the length k.
        divisors = []
        for k_candidate in range(1, int(math.sqrt(n)) + 1):
            if n % k_candidate == 0:
                divisors.append(k_candidate)
                # Add the corresponding divisor pair if it's different (i.e., k_candidate is not the square root of n).
                if k_candidate * k_candidate != n:
                    divisors.append(n // k_candidate)
        
        # Sort the divisors in ascending order to ensure we find the minimum k first.
        divisors.sort() 

        # Iterate through the sorted divisors.
        for k in divisors:
            # First check the necessary condition: k must be a multiple of required_lcm.
            # This potentially filters out many divisors, speeding up the process.
            if k % required_lcm == 0:
                # If the necessary condition holds, perform the sufficient check:
                # Verify if all segments of length k in s are indeed anagrams of the first segment.
                # The _check_k function performs this verification.
                if Solution._check_k(s, n, k): 
                    # Since we are iterating through divisors in increasing order,
                    # the first divisor k that satisfies both conditions is the minimum possible length.
                    return k
        
        # This line should theoretically be unreachable given the problem constraints (n >= 1).
        # The length n itself is always a valid divisor, n % required_lcm == 0 will hold, 
        # and _check_k(s, n, n) is always true (t=s case).
        # It serves as a fallback.
        return n