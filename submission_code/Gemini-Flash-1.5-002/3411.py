class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        big_nums = []
        def powerful_array(x):
            res = []
            i = 0
            while x > 0:
                power_of_two = 1 << i
                if x >= power_of_two:
                    res.append(power_of_two)
                    x -= power_of_two
                i += 1
            return res
        
        num_count = 0
        for i in range(1,10**6+1):
            num_count += len(powerful_array(i))
            if num_count > 10**15:
                break
        
        big_nums_map = {}
        count = 0
        for i in range(1,10**6+1):
            powerful_arr = powerful_array(i)
            for num in powerful_arr:
                big_nums_map[count] = num
                count += 1
                if count > 10**15:
                    break
            if count > 10**15:
                break

        ans = []
        for query in queries:
            from_i, to_i, mod_i = query
            product = 1
            for i in range(from_i-1, to_i):
                product = (product * big_nums_map[i]) % mod_i
            ans.append(product)
        return ans