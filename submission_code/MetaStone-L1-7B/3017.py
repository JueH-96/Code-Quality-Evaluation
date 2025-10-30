class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        beautiful_numbers = self.generate_beautiful_numbers()
        count = 0
        for num in beautiful_numbers:
            if low <= num <= high and num % k == 0:
                count += 1
        return count
    
    def generate_beautiful_numbers(self):
        beautiful_numbers = []
        def backtrack(current_num, current_length, even, odd):
            if current_length % 2 == 0 and even == odd:
                beautiful_numbers.append(current_num)
            if current_length >= 10:
                return
            if current_length == 0:
                for d in range(1, 10):
                    new_num = d
                    new_length = 1
                    new_even = 1 if d % 2 == 0 else 0
                    new_odd = 1 if d % 2 == 1 else 0
                    backtrack(new_num, new_length, new_even, new_odd)
            else:
                for d in range(0, 10):
                    new_num = current_num * 10 + d
                    new_length = current_length + 1
                    new_even = even + (d % 2 == 0)
                    new_odd = odd + (d % 2 == 1)
                    backtrack(new_num, new_length, new_even, new_odd)
        backtrack(0, 0, 0, 0)
        return beautiful_numbers