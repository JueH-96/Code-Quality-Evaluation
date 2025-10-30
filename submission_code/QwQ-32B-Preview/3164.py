class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        integers = []
        result = []
        consecutive_prev = 0
        
        for word in words:
            if word.isdigit():  # Check if the word is an integer
                integers.append(int(word))
                consecutive_prev = 0  # Reset the consecutive_prev counter
            else:
                consecutive_prev += 1  # Increment for "prev"
                k = consecutive_prev
                nums_reverse = integers[::-1]  # Reverse the integers list
                if k - 1 < len(nums_reverse):
                    result.append(nums_reverse[k - 1])
                else:
                    result.append(-1)
        return result