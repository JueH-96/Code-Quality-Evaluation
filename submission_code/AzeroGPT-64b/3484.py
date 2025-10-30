class Solution:
    def getSmallestString(self, s: str) -> str:
        even = []
        odd = []
        
        # Separate even and odd digits
        for i, val in enumerate(s):
            if int(val) % 2 == 0:
                even.append((val, i))
            else:
                odd.append((val,i))
        
        # Sort the pairs of even and odd digits in lexicographical order
        even.sort(key=lambda x: x[0])
        odd.sort(key=lambda x: x[0])
        
        swaps = 0
        s_list = list(s)
        
        # Sort the string by placing the smallest even digits in their original places first
        for digit, index in even:
            s_list[index] = digit
            swaps += 1
            if swaps > 1:
                break
        
        # Then place the smallest odd digits in their original places
        for digit, index in odd:
            s_list[index] = digit
            swaps += 1
            if swaps > 1:
                break
        
        return "".join(s_list)