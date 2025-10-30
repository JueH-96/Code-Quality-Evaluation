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
        
        def get_big_nums_index(index):
            low = 1
            high = 10**18
            while low < high:
                mid = (low + high) // 2
                total = 0
                for i in range(1, mid+1):
                    total += len(get_powerful_array(i))
                    if total > index:
                        break
                if total >= index:
                    high = mid
                else:
                    low = mid + 1
            x = low
            total_before = 0
            for i in range(1, x):
                total_before += len(get_powerful_array(i))
            pos = index - total_before
            powerful = get_powerful_array(x)
            return powerful[pos-1]
        
        def compute_product(from_i, to_i, mod):
            product = 1
            for i in range(from_i, to_i+1):
                num = get_big_nums_index(i)
                product = (product * num) % mod
            return product
        
        answer = []
        for query in queries:
            from_i, to_i, mod_i = query
            result = compute_product(from_i, to_i, mod_i)
            answer.append(result)
        return answer