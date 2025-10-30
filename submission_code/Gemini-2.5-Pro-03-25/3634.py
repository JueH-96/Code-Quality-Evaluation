import collections

class Solution:
  """
  Solves the problem of calculating a score based on finding mirror character pairs in a string.
  The process involves iterating through the string and, for each character, finding the closest
  preceding unmarked character that is its mirror image in the reversed alphabet.
  If such a pair is found, both characters are marked, and the distance between their indices
  is added to the score.
  """
  def calculateScore(self, s: str) -> int:
    """
    Calculates the score by iterating through the string and matching characters
    with their mirrors found at earlier indices.

    Args:
        s: The input string consisting of lowercase English letters. 
           Constraints: 1 <= s.length <= 10^5.

    Returns:
        The total calculated score based on the described process.
        The score is the sum of (i - j) for all matched pairs (i, j) where j < i.
    """
    
    # Precompute the mirror map for lowercase English letters.
    # The mirror of 'a' is 'z', 'b' is 'y', ..., 'z' is 'a'.
    # This can be calculated using ASCII values: mirror(c) = chr(ord('a') + ord('z') - ord(c)).
    mirror_map = {}
    sum_a_z = ord('a') + ord('z') # Constant value 219 for ASCII lowercase letters
    for i in range(26):
        char_code = ord('a') + i
        char = chr(char_code)
        mirror_code = sum_a_z - char_code
        mirror_char = chr(mirror_code)
        mirror_map[char] = mirror_char

    # Use a dictionary to keep track of the indices of unmarked characters encountered so far.
    # The keys are characters, and the values are lists of indices where that character appears
    # and is currently unmarked. These lists act like stacks, where the last element added
    # is the most recent (largest index) occurrence of that character.
    # collections.defaultdict(list) initializes an empty list for any key accessed for the first time.
    unmarked_indices = collections.defaultdict(list)
    
    score = 0
    n = len(s)

    # Iterate through the string from left to right, using index i.
    for i in range(n):
        char = s[i]
        # Determine the mirror character required for a match at index i.
        mirror_char = mirror_map[char]

        # Check if there is any available (unmarked) index j for the mirror character
        # such that j < i. The list unmarked_indices[mirror_char] stores such indices
        # encountered so far, in increasing order.
        # If the list is not empty, its last element corresponds to the largest such index j,
        # which is the closest one to i according to the problem statement.
        if unmarked_indices[mirror_char]:
            # A match is found. Get the index j by popping the last element from the list.
            # This ensures we use the closest unmarked mirror index j < i.
            j = unmarked_indices[mirror_char].pop() 
            
            # Add the difference (i - j) to the total score.
            score += i - j
            
            # By popping j from the list, we effectively mark index j as used.
            # Index i is also considered marked because it formed a pair with j.
            # We do not add index i to any list because it's now marked.
        else:
            # No suitable unmarked mirror character was found at any index j < i.
            # The current character at index i remains unmarked for now.
            # It becomes available as a potential match for characters at later indices k > i.
            # Add the current index i to the list associated with its character s[i].
            unmarked_indices[char].append(i)

    # After iterating through the entire string, return the final calculated score.
    return score