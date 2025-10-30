class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        counter = list(s)
        lastOne = counter.pop(counter.index('1')) # Get the last one and delete it from the list
        counter.sort(reverse=True) # sort the rest of the digits
        counter.append(lastOne) # Put the last one at the end of the string
        
        return "".join(counter)