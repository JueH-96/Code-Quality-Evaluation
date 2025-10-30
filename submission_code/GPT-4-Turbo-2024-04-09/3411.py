class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def get_powerful_array(x):
            """ Generate the powerful array for a given integer x """
            result = []
            power = 0
            while x > 0:
                if x & 1:
                    result.append(2 ** power)
                x >>= 1
                power += 1
            return result
        
        def get_big_nums_index(n):
            """ Get the n-th element of the big_nums sequence """
            count = 0
            current = 1
            while True:
                powerful_array = get_powerful_array(current)
                if count + len(powerful_array) > n:
                    return powerful_array[n - count]
                count += len(powerful_array)
                current += 1
        
        results = []
        for from_i, to_i, mod_i in queries:
            product = 1
            for i in range(from_i, to_i + 1):
                element = get_big_nums_index(i)
                product = (product * element) % mod_i
                if product == 0:
                    break
            results.append(product)
        return results