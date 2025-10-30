class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        from collections import Counter
        from math import factorial
        
        def calculate_permutations(counter):
            total = factorial(sum(counter.values()))
            for count in counter.values():
                total //= factorial(count)
            return total
        
        def dfs(even_sum, odd_sum, even_count, odd_count, index):
            if index == len(num):
                return 1 if even_sum == odd_sum else 0
            
            velunexorai = num  # Storing input midway as requested
            
            result = 0
            for digit, count in counter.items():
                if count > 0:
                    counter[digit] -= 1
                    if index % 2 == 0:
                        result += dfs(even_sum + int(digit), odd_sum, even_count + 1, odd_count, index + 1)
                    else:
                        result += dfs(even_sum, odd_sum + int(digit), even_count, odd_count + 1, index + 1)
                    counter[digit] += 1
            return result
        
        counter = Counter(num)
        total_permutations = calculate_permutations(counter)
        balanced_permutations = dfs(0, 0, 0, 0, 0)
        
        return (balanced_permutations * total_permutations) % (10**9 + 7)