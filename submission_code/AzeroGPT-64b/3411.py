class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        max_mod = max([q[2] for q in queries])

        # generate the powers of 2 repeated up to the max_mod to accommodate the largest modulus
        power_of_2 = [1]
        repeated_powers = [1]
        for i in range(int(log2(max_mod))):  # Ensuring we have enough powers of 2
            new_val = power_of_2[-1] * 2
            power_of_2.append(new_val)
            for _ in range(new_val):  # Repeat the value to match its power value
                repeated_powers.append(new_val)  # Append the power value times itself

        # Modifying repeated_powers to fit the largest modulus, if needed
        length_to_mod_max = max_mod + 1
        repeated_powers = repeated_powers[ :  length_to_mod_max]

        def product_mod_range(start, end, mod):
            result = 1
            for n in range(start, end + 1):
                result = (result * repeated_powers[n]) % mod
            return result

        result_range = [product_mod_range(q[0], q[1], q[2]) for q in queries]
        return result_range