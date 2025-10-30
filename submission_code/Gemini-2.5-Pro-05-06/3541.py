from typing import List

class Solution:
  def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
    """
    Checks if a message is spam based on a list of banned words.
    A message is spam if at least two words in it exactly match any word in bannedWords.
    """
    
    # For efficient lookup of banned words, convert the list of bannedWords
    # into a set. The average time complexity for checking membership
    # in a set is O(1) (amortized, after considering the time to hash the word itself, 
    # which is proportional to word length).
    # Creation of the set takes O(sum of lengths of words in bannedWords).
    banned_set = set(bannedWords)
    
    # Counter for the number of words from the 'message' array
    # that are found in the 'banned_set'.
    banned_occurrences_in_message = 0
    
    # Iterate through each word in the 'message' array.
    # This loop runs up to 'len(message)' times.
    for word in message:
      # Check if the current 'word' from 'message' is in our 'banned_set'.
      # This check takes time proportional to the length of 'word' on average.
      if word in banned_set:
        banned_occurrences_in_message += 1
        
        # If we have found two or more banned words in the 'message' array,
        # the condition for spam is met. We can stop processing and return True.
        if banned_occurrences_in_message >= 2:
          return True
          
    # If the loop completes, it means 'banned_occurrences_in_message' is either 0 or 1.
    # In this case, the message is not spam according to the definition.
    return False