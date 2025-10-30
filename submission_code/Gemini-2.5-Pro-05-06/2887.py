class Solution:
  def sortVowels(self, s: str) -> str:
    vowel_set = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
    
    extracted_vowels = []
    vowel_indices = [] # Store indices of vowels in the original string
    
    # Pass 1: Collect vowels and their original indices
    for i, char_val in enumerate(s):
        if char_val in vowel_set:
            extracted_vowels.append(char_val)
            vowel_indices.append(i)
    
    # Sort the collected vowels by ASCII value
    # Python's default sort for characters in a list is lexicographical,
    # which for single characters means sorting by their ASCII values.
    extracted_vowels.sort()
    
    # Convert the string to a list of characters to allow modification,
    # as strings are immutable in Python.
    result_list = list(s) 
    
    # Pass 2: Place sorted vowels back into their original positions in the string
    for i in range(len(extracted_vowels)):
        # Get the original index of the i-th vowel slot
        idx_in_s = vowel_indices[i] 
        # Get the i-th vowel from the sorted list
        sorted_vowel_char = extracted_vowels[i]
        # Place the sorted vowel into the result list at its original position
        result_list[idx_in_s] = sorted_vowel_char
            
    # Join the list of characters back into a string
    return "".join(result_list)