class Solution:
  def makeSmallestPalindrome(self, s: str) -> str:
    n = len(s)
    # Convert string to list of characters because strings are immutable in Python.
    # This allows us to modify characters in place.
    arr = list(s) 
    
    left = 0
    right = n - 1
    
    # Iterate from both ends of the string towards the center.
    # The loop continues as long as the left pointer is to the left of the right pointer.
    # When left >= right, all pairs have been processed, or we are at the middle character 
    # (if N is odd), which doesn't need modification for palindromic property with another character.
    while left < right:
      # If the characters at the current pair of positions (left and right) are different.
      if arr[left] != arr[right]:
        # We need to make them equal. To satisfy the palindrome condition,
        # the character at arr[left] must be equal to the character at arr[right] in the final palindrome.
        # This will require one operation (changing one of the characters).
        # This is the minimum number of operations for this pair.
        
        # To ensure the resulting palindrome is lexicographically the smallest,
        # we choose the smaller of the two characters (arr[left] or arr[right])
        # to be the character at both positions.
        # The character arr[left] (at the earlier index) is prioritized for
        # lexicographical comparison, so it should be as small as possible.
        if arr[left] < arr[right]:
          # If arr[left] is already the smaller character, we change arr[right]
          # to match arr[left]. arr[left] remains unchanged.
          arr[right] = arr[left]
        else:
          # If arr[right] is smaller (or they would have been equal, but that's 
          # caught by the `arr[left] != arr[right]` condition),
          # we change arr[left] to match arr[right]. arr[right] remains unchanged.
          # This makes arr[left] (and consequently arr[right]) the smaller of the two original characters.
          arr[left] = arr[right]
      
      # Move the pointers one step closer to the center for the next iteration.
      left += 1
      right -= 1
      
    # Convert the list of characters back to a string and return it.
    return "".join(arr)