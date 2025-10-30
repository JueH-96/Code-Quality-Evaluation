class Solution:
    def smallestNumber(self, n: int) -> int:
        binary_n = bin(n)[2:]
        index_of_zero = -1
        for i in range(len(binary_n)):
            if binary_n[i] == '0':
                index_of_zero = i
                break
        if index_of_zero == -1:
            return n
        else:
            length = len(binary_n)
            all_ones_binary = '1' * length
            return int(all_ones_binary, 2)