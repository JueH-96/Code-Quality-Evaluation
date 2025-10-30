class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        """
        The function determines the number of balanced permutations, which means that the sum of the elements at even and odd indices is equal.
        The result is returned modulo 1,000,000,007 for large numbers.
        """
        even_digit, odd_digit = 0, 0
        # This dictionary will keep track of the count of each digit number.
        store_digit = {}
        
        for index, value in enumerate(num):
            if index % 2 == 0:
                even_digit += int(value)
            else:
                odd_digit += int(value)
            store_digit[value] = store_digit.setdefault(value, 0) + 1
        
        # If the total sum of digits at even and odd indices is not equal, no permutation can be balanced.
        if even_digit != odd_digit:
            return 0
        
        # Calculate the total number of ways to choose which digits go into an even or odd position for each digit number. 
        even_total, odd_total, answer = 1, 1, 1
        for i in store_digit.values():
            # Combination formula to choose position.
            even_total *= self.combination(i, len(num) // 2 - len(store_digit) + i)
            even_total %= 1000000007
                
            odd_total *= self.combination(i, len(num) // 2 - len(store_digit) + i)
            odd_total %= 1000000007
            
        return (even_total * odd_total) % 1000000007
    
    def combination(self, n, k):
        """
        Helper function to calculate combinations using a Fast Modulo Multiplicative Inverse.
        It is more efficient for large numbers because it avoids calculating large intermediate values.
        """
        if k > n: return 0
        nFact, kFact, n_kFact = 1, 1, 1
        for i in range(1, n+1):
            nFact = (nFact * i) % 1000000007
            if i <= k:
                kFact = (kFact * i) % 1000000007
            if i <= (n - k):
                n_kFact = (n_kFact * i) % 1000000007
        
        # Calculating the result using the Modular Multiplicative Inverse.
        return (nFact * self.modInverse(kFact, 1000000007) * 
           self.modInverse(n_kFact, 1000000007)) % 1000000007
    
    def modInverse(self, n, p):
        """
        Helper function to calculate Modular Multiplicative Inverse using Fermat's Little Theorem.
        """
        return pow(n, p-2, p)