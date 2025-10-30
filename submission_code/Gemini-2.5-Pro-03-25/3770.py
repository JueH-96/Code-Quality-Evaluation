import sys
# Setting a higher recursion depth is generally not necessary for iterative KMP.
# sys.setrecursionlimit(2000) 

class Solution:
    
    def compute_kmp_pi(self, pattern: str) -> list[int]:
        """
        Computes the KMP failure function (pi table) for a given pattern.
        The pi table stores for each index q, the length of the longest proper prefix 
        of pattern[0..q] that is also a suffix of pattern[0..q].
        """
        m = len(pattern)
        # Base case: empty pattern has empty pi table
        if m == 0:
             return []
        
        pi = [0] * m # Initialize pi table with zeros
        k = 0 # k represents the length of the previous longest prefix suffix match
        
        # Compute pi[q] for q from 1 to m-1 (pi[0] is always 0)
        for q in range(1, m):
            # While k > 0 (meaning there is a potential prefix/suffix match to extend or backtrack from)
            # and the character following the current prefix match (pattern[k]) 
            # does not match the current character pattern[q],
            # we need to find the next shorter prefix that is also a suffix.
            # We use the pi table itself to efficiently find this shorter length: k = pi[k-1].
            while k > 0 and pattern[k] != pattern[q]:
                k = pi[k - 1]
            
            # If the character following the prefix (pattern[k]) matches the current character pattern[q]
            if pattern[k] == pattern[q]:
                k += 1 # Extend the length of the current prefix/suffix match
            
            # Store the computed length k (length of longest prefix/suffix ending at q) into pi[q]
            pi[q] = k
            
        return pi

    def generateString(self, str1: str, str2: str) -> str:
        """
        Generates the lexicographically smallest string 'word' of length n + m - 1
        satisfying the constraints defined by str1 and str2.
        str1[i] == 'T' implies word[i:i+m] == str2
        str1[i] == 'F' implies word[i:i+m] != str2
        Returns "" if no such string exists.
        """
        n = len(str1)
        m = len(str2)
        
        # Constraints state m >= 1, so no need to handle m=0 explicitly.
        
        L = n + m - 1 # Length of the target string 'word'

        # Phase 1: Pre-computation based on 'T' constraints.
        # Determine required characters at each position k due to str1[i] == 'T'.
        # Check for conflicts where multiple 'T' constraints require different characters at the same position.
        req_char = [None] * L # Stores the required character at each position k, if any.
        possible = True # Flag to track if 'T' constraints are consistent.
        for i in range(n):
            if str1[i] == 'T':
                # If str1[i] == 'T', the substring word[i : i+m] must equal str2.
                # This imposes constraints on characters word[i] through word[i+m-1].
                for j in range(m):
                    k = i + j # k is the index in 'word'
                    
                    # Ensure k is a valid index within the bounds of 'word' (0 to L-1).
                    # Based on L = n + m - 1, the maximum k is (n-1) + (m-1) = n+m-2 = L-1,
                    # so k will always be less than L. No check needed for k >= L.

                    required = str2[j] # The character required at word[k] by this constraint.
                    
                    # Check if position k already has a required character assigned.
                    if req_char[k] is None:
                        # If not, assign the required character.
                        req_char[k] = required
                    elif req_char[k] != required:
                        # If already assigned and it conflicts with the new requirement, then it's impossible.
                        possible = False
                        break # Conflict found, stop checking this 'T' constraint (inner loop).
                if not possible:
                    break # Conflict found, stop checking all 'T' constraints (outer loop).
        
        # If pre-computation revealed an impossible scenario (conflicting 'T' constraints).
        if not possible:
            return ""

        # Phase 2: Greedy construction of 'word' with KMP checks for 'F' constraints.
        
        # Precompute the KMP failure function (pi table) for str2.
        # This will help efficiently check if placing a character completes a match of str2,
        # which is needed for verifying 'F' constraints.
        pi = self.compute_kmp_pi(str2)
        
        # Initialize the result string builder (using a list for mutable characters).
        word = [''] * L
        # Initialize KMP state `q`. `q` tracks the length of the longest prefix of `str2`
        # that is a suffix of the `word` prefix constructed so far (word[0...k-1]).
        q = 0 # Initial state is 0 (empty prefix match).

        # Iterate through each position `k` of the `word` string from 0 to L-1.
        for k in range(L):
            found_char = False # Flag: Did we find a valid character for word[k]?
            
            # Determine the range of characters to try for word[k].
            # If req_char[k] is specified by a 'T' constraint, only try that character.
            # Otherwise, try 'a' through 'z' to find the lexicographically smallest valid one.
            start_char_ord = ord('a')
            end_char_ord = ord('z')
            if req_char[k] is not None:
                start_char_ord = ord(req_char[k])
                end_char_ord = ord(req_char[k])

            # Iterate through possible characters `c` for `word[k]` in lexicographical order.
            for c_ord in range(start_char_ord, end_char_ord + 1):
                c = chr(c_ord)
                
                # Check 1 (Consistency with 'T' constraints) is implicitly handled:
                # The loop range already ensures we only consider characters consistent with req_char[k].

                # Check 2 (Consistency with 'F' constraints): Use KMP state transitions.
                # Calculate the potential next KMP state `q_prime_k` if we append character `c`.
                
                # `current_q` starts from the state `q` achieved after processing `word[k-1]`.
                current_q = q 
                
                # Handle KMP state reset logic: If the previous state `q` was `m` (a full match ended at k-1),
                # the state for considering character `c` should start from the length of the border of str2.
                # Need to check m > 0 before accessing pi table or str2.
                if m > 0 and current_q == m: 
                     current_q = pi[m-1] # Reset state based on the border length

                # Perform the standard KMP state transition using character `c`.
                # Find the longest prefix of `str2` that is a suffix of `word[0...k-1] + c`.
                # Need m > 0 check for accessing str2.
                while m > 0 and current_q > 0 and c != str2[current_q]:
                    current_q = pi[current_q - 1]
                
                # If `c` matches the character at `str2[current_q]`, we extend the current prefix match.
                # Need m > 0 and current_q < m checks for safe access to str2[current_q].
                if m > 0 and current_q < m and c == str2[current_q]: 
                    current_q += 1
                
                # `q_prime_k` is the resulting KMP state after processing character `c` at index `k`.
                q_prime_k = current_q 

                # Check if this new state `q_prime_k == m` implies a completed match of `str2`
                # that is forbidden by an 'F' constraint.
                is_forbidden = False
                if q_prime_k == m: # A full match of `str2` ends at index `k`.
                    # The match spans `word[k - m + 1 : k + 1]`.
                    i = k - m + 1 # Starting index `i` of this match in `word`.
                    
                    # Check if there is an 'F' constraint at this starting index `i`.
                    # Ensure `i` is a valid index for str1 (0 <= i < n).
                    if 0 <= i < n and str1[i] == 'F':
                        # If yes, then placing character `c` creates a substring match forbidden by str1[i] == 'F'.
                        is_forbidden = True
                
                # If character `c` is not forbidden by any 'F' constraint:
                if not is_forbidden:
                    # We found the lexicographically smallest valid character for `word[k]`.
                    word[k] = c
                    q = q_prime_k # Update the KMP state for the next iteration (k+1).
                    found_char = True
                    break # Exit the character loop (`for c_ord ...`) and proceed to the next position `k+1`.
            
            # If the character loop completed without finding any valid character for `word[k]`:
            if not found_char:
                # It is impossible to construct the string satisfying all constraints with the prefix built so far.
                # Since we always pick the smallest valid character greedily, this means no solution exists overall.
                return ""

        # If all positions `k` from 0 to `L-1` were successfully filled:
        return "".join(word) # Concatenate the list of characters into the final result string.