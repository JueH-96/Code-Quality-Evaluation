class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def get_price(num, x):
            price = 0
            binary = bin(num)[2:]
            for i in range(len(binary)):
                if (len(binary) - i) % x == 0 and binary[i] == '1':
                    price += 1
            return price

        def check(num, k, x):
            total_price = 0
            for i in range(1, num + 1):
                total_price += get_price(i, x)
            return total_price <= k
        
        def get_price_optimized(num, x):
            price = 0
            power_of_2 = 1
            for i in range(1, 65):
                if i % x == 0:
                    
                    full_cycles = num // (power_of_2 * 2)
                    price += full_cycles * power_of_2
                    
                    remaining = num % (power_of_2 * 2)
                    price += max(0, remaining - power_of_2 + 1)
                power_of_2 *= 2
            return price

        def check_optimized(num, k, x):
            return get_price_optimized(num, x) <= k

        left, right = 1, 2**55
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check_optimized(mid, k, x):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans