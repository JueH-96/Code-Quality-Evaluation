import math

class Solution:
    """
    Finds the lexicographically largest string that can appear as a piece 
    when splitting the word into numFriends non-empty strings.

    The core idea is to determine the set of all possible strings that can
    result from any valid split, and then find the lexicographically largest
    string within that set.

    A split of `word` into `numFriends` non-empty strings `s_1, s_2, ..., s_numFriends`
    means `word = s_1 + s_2 + ... + s_numFriends`, where each `s_i` is non-empty.

    Consider a substring `sub = word[i:j]`. Can this substring be one of the pieces `s_k`
    in some valid split?
    If `sub` is the k-th piece, the prefix `word[0:i]` must be split into `k-1` non-empty
    pieces, and the suffix `word[j:n]` must be split into `numFriends - k` non-empty pieces.
    This requires `len(word[0:i]) >= k-1` (i.e., `i >= k-1`) and 
    `len(word[j:n]) >= numFriends - k` (i.e., `n - j >= numFriends - k`).
    Summing these conditions gives `i + n - j >= numFriends - 1`.
    Rearranging, `n - (j - i) >= numFriends - 1`.
    Let `L = j - i` be the length of the substring `sub`. Then `n - L >= numFriends - 1`.
    This simplifies to `L <= n - numFriends + 1`.

    So, a substring `sub` can be a piece in a valid split if and only if its length `L`
    satisfies `1 <= L <= n - numFriends + 1`. Let `max_len = n - numFriends + 1`.

    The problem reduces to finding the lexicographically largest substring `word[i:j]`
    such that its length `j - i` is between 1 and `max_len` (inclusive).

    We can iterate through all possible substrings `word[i:j]` satisfying the length constraint
    and keep track of the maximum one found. However, this can be optimized.

    Consider all valid substrings starting at a fixed index `i`. These are `word[i : i+L]`
    where `1 <= L <= min(max_len, n - i)`. Let this upper bound on L be `L_max_i`.
    The set of candidates starting at `i` is `{ word[i:i+1], word[i:i+2], ..., word[i:i+L_max_i] }`.
    Since `word[i:i+k]` is a proper prefix of `word[i:i+k+1]`, we know that
    `word[i:i+k+1]` is lexicographically greater than `word[i:i+k]`.
    Therefore, the lexicographically largest substring starting at `i` and satisfying the
    length constraint is the longest one: `word[i : i + L_max_i]`.

    The overall lexicographically largest substring must be the maximum among these 
    "best candidates" for each starting position `i`.
    So, we need to compute `max { word[i : i + min(max_len, n - i)] | 0 <= i < n }`.
    This can be done by iterating through `i` from 0 to `n-1`, calculating the candidate
    substring for each `i`, and comparing it with the current maximum found so far.
    This approach has a time complexity of O(n^2) due to the loop and string operations 
    (slicing and comparison), which is feasible for n up to 5000.
    """
    def answerString(self, word: str, numFriends: int) -> str:
        """
        Args:
            word: The input string. Length n >= 1.
            numFriends: The number of friends (pieces to split into). 1 <= numFriends <= n.

        Returns:
            The lexicographically largest possible piece string.
        """
        n = len(word)
        
        # Calculate the maximum allowed length for a piece.
        # A substring word[i:j] (length L = j-i) can be a piece if the remaining 
        # n - L characters can form numFriends - 1 non-empty pieces.
        # This requires n - L >= numFriends - 1, so L <= n - numFriends + 1.
        max_len = n - numFriends + 1
        
        # Initialize max_string with the best candidate starting at i = 0.
        # The length of this candidate is L_0 = min(max_len, n - 0) = min(max_len, n).
        # Since n >= 1 and numFriends <= n, max_len = n - numFriends + 1 >= 1.
        # Thus, L_0 >= 1.
        current_best_len = min(max_len, n)
        max_string = word[0 : current_best_len]
        
        # Iterate through the remaining starting positions i = 1 to n-1.
        for i in range(1, n):
            # Calculate the length of the best candidate substring starting at i.
            # L_i = min(max_len, n - i).
            # Since i < n, n - i >= 1. Since max_len >= 1, L_i >= 1.
            current_len = min(max_len, n - i)
            
            # Extract the candidate substring M_i = word[i : i + L_i].
            # Python slicing word[start:end] extracts characters from index start up to, but not including, end.
            candidate_sub = word[i : i + current_len]
            
            # Compare the candidate M_i with the current maximum string found so far.
            # Python's string comparison performs lexicographical comparison.
            if candidate_sub > max_string:
                max_string = candidate_sub
                
        return max_string