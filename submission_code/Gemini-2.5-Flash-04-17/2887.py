class Solution:
    def sortVowels(self, s: str) -> str:
        # Define the set of vowels for efficient checking
        vowels = set("aeiouAEIOU")
        
        # 1. Collect all vowel characters from the input string
        vowel_chars = []
        for char in s:
            if char in vowels:
                vowel_chars.append(char)
                
        # 2. Sort the collected vowel characters based on their ASCII values
        # Python's default sort for strings/chars uses ASCII order
        vowel_chars.sort()
        
        # 3. Create a mutable list representation of the input string
        # We will modify this list in place
        result_list = list(s)
        
        # 4. Iterate through the result list. When a vowel position is found,
        # replace the original vowel with the next vowel from the sorted list.
        
        # Initialize a pointer for the sorted vowel_chars list
        vowel_idx = 0 
        
        for i in range(len(result_list)):
            # Check if the character at the current index is a vowel
            if result_list[i] in vowels:
                # If it's a vowel, replace it with the next available sorted vowel
                result_list[i] = vowel_chars[vowel_idx]
                # Move the pointer to the next sorted vowel
                vowel_idx += 1 
                
        # 5. Join the modified list back into a string
        return "".join(result_list)