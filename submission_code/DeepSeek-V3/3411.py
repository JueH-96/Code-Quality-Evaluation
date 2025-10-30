class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def get_powerful_array(x):
            res = []
            power = 1
            while x > 0:
                if x & 1:
                    res.append(power)
                power <<= 1
                x >>= 1
            return res
        
        def get_big_nums_index(n):
            low, high = 1, 10**18
            while low < high:
                mid = (low + high) // 2
                total = 0
                i = 1
                while True:
                    cnt = mid // (1 << i)
                    if cnt == 0:
                        break
                    total += cnt
                    i += 1
                if total < n:
                    low = mid + 1
                else:
                    high = mid
            return low
        
        def get_big_nums_segment(start, end):
            res = []
            current = start
            while current <= end:
                x = get_big_nums_index(current)
                arr = get_powerful_array(x)
                for num in arr:
                    if current > end:
                        break
                    res.append(num)
                    current += 1
            return res
        
        answer = []
        for query in queries:
            from_i, to_i, mod_i = query
            segment = get_big_nums_segment(from_i + 1, to_i + 1)
            product = 1
            for num in segment:
                product = (product * num) % mod_i
            answer.append(product)
        return answer