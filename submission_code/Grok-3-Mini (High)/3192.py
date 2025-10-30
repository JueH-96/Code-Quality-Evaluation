class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 1000000007
        P = 1 << n
        A_high = a // P
        B_high = b // P
        D = (a ^ b) & ((1 << n) - 1)
        A_acc = A_high * P
        B_acc = B_high * P
        
        for i in range(n - 1, -1, -1):
            if (D >> i) & 1 == 0:  # D_i == 0
                A_acc += (1 << i)
                B_acc += (1 << i)
            else:  # D_i == 1
                if A_acc <= B_acc:
                    A_acc += (1 << i)
                else:
                    B_acc += (1 << i)
        
        return (A_acc * B_acc) % MOD