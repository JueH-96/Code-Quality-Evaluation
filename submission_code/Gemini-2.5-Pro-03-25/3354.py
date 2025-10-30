import heapq

class Solution:
    """
    Solves the problem of replacing '?' in a string s to minimize its value,
    returning the lexicographically smallest result among optimal strings.
    The value of a string t is the sum of cost(i) for all i, where cost(i)
    is the number of characters equal to t[i] appearing before index i.
    The total value of a string is equivalent to sum_{c='a'}^{'z'} k_c*(k_c-1)/2,
    where k_c is the total count of character c in the string.
    Minimizing this value involves distributing the '?' replacements such that
    the final counts k_c are as balanced as possible. This is achieved by
    greedily assigning each '?' to the character type that currently has the
    minimum total count. Ties are broken by choosing the lexicographically
    smallest character.
    To ensure the final string is lexicographically smallest among optimal solutions,
    the determined replacement characters are sorted and assigned to the '?'
    positions from left to right.
    """
    def minimizeStringValue(self, s: str) -> str:
        """
        Replaces '?' characters in s to minimize string value and returns the
        lexicographically smallest result.

        Args:
          s: The input string containing lowercase letters and '?'.

        Returns:
          The modified string with '?' replaced, having minimum possible value
          and being lexicographically smallest among minimum-value strings.
        """
        
        n = len(s)
        # initial_counts array stores the counts of each character 'a' through 'z'
        # based only on the fixed characters initially present in s.
        initial_counts = [0] * 26 
        # q_indices list stores the indices where s[i] == '?'. The order is preserved.
        q_indices = []    
        
        # Pass 1: Calculate initial counts from non-'?' characters and identify '?' indices.
        for i in range(n):
            if s[i] == '?':
                q_indices.append(i)
            else:
                # Increment count for the existing character s[i]
                initial_counts[ord(s[i]) - ord('a')] += 1
        
        # Total number of '?' characters that need replacement.
        Q = len(q_indices)
        
        # If there are no '?', the string is already fixed and optimal. Return it.
        if Q == 0:
            return s 
        
        # replacement_q_counts stores how many times each character 'a' through 'z'
        # is chosen to replace a '?'. This distribution minimizes the total value.
        # Initialize counts to zero.
        replacement_q_counts = [0] * 26 
        
        # Min-heap stores tuples (current_total_count, char_index) for each character type.
        # The heap property is based first on current_total_count, then on char_index.
        # This allows efficiently finding the character with minimum current total count.
        # Ties in count are broken by char_index (lower index means lexicographically smaller character).
        min_heap = []
        for i in range(26):
            # Push initial state onto the heap based on fixed characters counts.
            heapq.heappush(min_heap, (initial_counts[i], i))
            
        # Pass 2: Greedily determine the counts of characters used for replacements.
        # This loop runs Q times, determining one replacement character in each iteration.
        for _ in range(Q):
            # Extract the character type (char_idx) with the minimum current total count (min_count).
            # heapq ensures that if counts are equal, the one with the smallest index (lexicographically first character) is popped.
            min_count, char_idx = heapq.heappop(min_heap)
            
            # Record that this character type (identified by char_idx) is used for one replacement.
            replacement_q_counts[char_idx] += 1
            
            # The new total count for this character type is its previous count + 1.
            new_total_count = min_count + 1
            # Push the updated state (new_total_count, char_idx) back into the heap.
            heapq.heappush(min_heap, (new_total_count, char_idx))
            
        # Pass 3: Construct the sorted list of replacement characters.
        # This list contains exactly Q characters needed to fill the '?'s.
        # The characters are generated in alphabetical order based on replacement_q_counts,
        # ensuring the list is sorted implicitly. This step takes O(Q + 26) = O(N) time.
        sorted_replacement_chars = []
        for i in range(26):
            # Character corresponding to index i ('a' + i).
            char = chr(ord('a') + i)
            # Append this character `replacement_q_counts[i]` times.
            # If replacement_q_counts[i] is 0, nothing is appended.
            sorted_replacement_chars.extend([char] * replacement_q_counts[i])
        
        # Pass 4: Create the final result string by filling in the '?' positions.
        # Convert s to a list of characters for mutable assignment. This takes O(N) time.
        s_list = list(s)
        
        # Fill the '?' positions using the sorted replacement characters.
        # The k-th '?' encountered (ordered by index) gets assigned the k-th character
        # from the sorted_replacement_chars list. This ensures the final string is
        # lexicographically smallest among those with minimum value. This loop runs Q times.
        for i in range(Q):
            # Index of the i-th '?' character in s.
            q_idx = q_indices[i]
            # The i-th character from the sorted list of replacement chars.
            replacement_char = sorted_replacement_chars[i]
            # Assign the character to the '?' position in the list representation of s.
            s_list[q_idx] = replacement_char
            
        # Join the list of characters back into the final string. This takes O(N) time.
        return "".join(s_list)