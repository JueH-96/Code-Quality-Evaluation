class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        count = 0
        distinct_numbers = set()
        n = len(digits)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and i != k and j != k:
                        num = digits[i] * 100 + digits[j] * 10 + digits[k]
                        if num >= 100 and num % 2 == 0:
                            distinct_numbers.add(num)
        
        return len(distinct_numbers)