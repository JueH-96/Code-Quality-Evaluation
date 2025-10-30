class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def calculate_price_sum(num, x):
            price_sum = 0
            for i in range(1, num + 1):
                binary = bin(i)[2:]
                for j in range(len(binary)):
                    if (len(binary) - j) % x == 0 and binary[j] == '1':
                        price_sum += 1
            return price_sum

        def calculate_price(num, x):
            price = 0
            binary = bin(num)[2:]
            for j in range(len(binary)):
                if (len(binary) - j) % x == 0 and binary[j] == '1':
                    price += 1
            return price
        
        low = 0
        high = 2**63 - 1
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            
            price_sum = 0
            for i in range(1, mid + 1):
                binary = bin(i)[2:]
                for j in range(len(binary)):
                    if (len(binary) - j) % x == 0 and binary[j] == '1':
                        price_sum += 1
            
            if price_sum <= k:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans