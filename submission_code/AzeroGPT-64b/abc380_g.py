from itertools import accumulate
from typing import List, Tuple

MOD = 998244353

class HistogramCalculator:
    """
    A class for calculating the weighted histogram of the sum of backward differences and the sum of forward differences
    in a list. This histogram is used for computing expected inversion numbers under certain operations on permutations.
    
    Attributes:
        N (int): The size of the list.
        K (Tuple[int]): Precomputed powers of K for MOD.
        invK (Tuple[int]): Precomputed modular inverses of powers of K for MOD.
    """
    
    def __init__(self, N: int):
        self.N = N
        self.K = [1] * (N + 1)
        self.invK = [1] * (N + 1)
        self.mod_comb = ModComb(MOD)
        for n in range(1, N + 1):
            self.K[n] = self.K[n - 1] * n % MOD
            self.invK[n] = self.invK[n - 1] * pow(n, MOD - 2, MOD) % MOD
    
    @staticmethod
    def count_inversions(A: List[int], N: int) -> int:
        """
        Count the number of inversions in a list.
        
        Parameters:
            A (List[int]): The list to count inversions in.
            N (int): The size of the list.
            
        Returns:
            int: The number of inversions.
        """
        B = [0] * N
        for i, a in enumerate(A):
            B[a - 1] = i
        ans = 0
        for i, b in enumerate(B):
            ans += b
            ans -= i
            ans -= B[i]
            for j in range(i + 1, N):
                if B[j] < B[i]:
                    B[j] += 1
                elif B[j] > B[i]:
                    B[j] -= 1
        return ans
    
    def build_histogram(self, A: List[int], K: int) -> Tuple[List[int], List[int]]:
        """
        Build a histogram of the sum of backward differences and the sum of forward differences in A.
        
        Parameters:
            A (List[int]): The list to build the histogram from.
            K (int): The number of elements to consider for the operations.
            
        Returns:
            Tuple[List[int], List[int]]: Two lists representing the histogram of backward and forward differences.
        """
        _factor = 1
        for _ in range(K - 1):
            _factor = _factor * self.N % MOD
        backW = [0] * (self.N - K + 1)
        forfW = [0] * (self.N - K + 1)
        curBack = _factor * (self.N - K) % MOD
        curForf = sum(A[:K - 1]) * _factor % MOD
        for i in range(self.N - K + 1):
            mask = [0] * self.N
            prefixSum = [0] * self.N
            for j in range(K, 0, -1):
                index = A[i + j - 1] - 1
                mask[index] = 1
            for j in range(self.N):
                prefixSum[j] = mask[j] + (prefixSum[j - 1] if j >= 1 else 0)
            for j in range(K):
                curBack -= prefixSum[A[i + j] - 1]
                curForf -= A[i + j] * prefixSum[j]
            backW[i] = curBack * self.invK[K - 1] % MOD
            forfW[i] = curForf * self.invK[K - 1] % MOD
            curBack = curBack * self.N % MOD
            curForf = curForf + self.N * (curForf % self.N) - A[i] * _factor % MOD
        return backW, forfW
    
    @staticmethod
    def sum_convolution(ans: List[int], A: List[int], B: List[int], MOD: int) -> List[int]:
        """
        Perform a convolution on two lists to calculate the sum of element-wise products.
        
        Parameters:
            ans (List[int]): The list to store the result in.
            A (List[int]): The first list.
            B (List[int]): The second list.
            MOD (int): The modulus for the operation.
            
        Returns:
            List[int]: The result of the convolution.
        """
        for r in range(len(A)):
            for l in range(len(B)):
                ans[r - l + len(B) - 1] += A[r] * B[l]
                ans[r - l + len(B) - 1] %= MOD
        return ans
    
    def calculate_expected_inversions(self, A: List[int], K: int) -> int:
        """
        Calculate the expected number of inversions under a specific operation on permutations.
        
        Parameters:
            A (List[int]): The permutation.
            K (int): The size of the segment to shuffle.
            
        Returns:
            int: The expected number of inversions modulo MOD.
        """
        ans = [0] * (K + 1)
        backW, forfW = self.build_histogram(A, K)
        factorial = [1] * (self.N + 1)
        invFact = [1] * (self.N + 1)
        for n in range(1, self.N + 1):
            factorial[n] = factorial[n - 1] * n % MOD
            invFact[n] = pow(factorial[n], MOD - 2, MOD)
        
        ans[0] = _factor = self.count_inversions(A, self.N)
        comb = [0] * (K + 1)
        for k in range(K + 1):
            comb[k] = factorial[K] * invFact[k] * invFact[K - k] % MOD
        for i in range(1, K + 1):
            temp = [0] * (self.N + 1)
            for l in range(K + 1):
                if l == 0:
                    temp[l] = comb[l] * pow(self.N, self.N - i - K + l)
                elif l == K:
                    temp[l] = comb[l] * self.N * pow(self.N - 1, self.N - i + l - K - 1) // pow(K, self.N - i + l - K) % MOD
                else:
                    t = comb[l] * pow(self.N, self.N - i - K + l)
                    for k in range(l + 1):
                        t *= self.invK[l - k]
                        t //= pow(l - k + 1, self.N - i + l - k) * pow(k + 1, self.N - i - k)
                    t *= self.K[l + 1]
                    temp[l] = t % MOD
            self.sum_convolution(ans, temp[:K], backW, MOD)
            for l in range(K + 1):
                _factor *= self.invK[l] * self.invK[K - l]
                _factor %= MOD
            for l in range(K + 1):
                temp[l] = ans[l] * _factor * invFact[l] * invFact[K - l] % MOD
            self.sum_convolution(ans, temp, forfW, MOD)
            ans[0] -= ans[K]
            ans[0] %= MOD
        ans[0] *=  self.K[K]
        ans[0] %= MOD
        return ans[0]

class ModComb:
    def __init__(self, mod):
        self.mod = mod
        self._fac, self._finv, self._inv = [1], [1], [1]
    
    def com_from_factorials(self, n, r):
        return self._fac[n] * self._finv[r] * self._finv[n-r] % self.mod
    
    def _make(self, n):
        fac, finv, inv = self._fac, self._finv, self._inv
        mod = self.mod
        for i in range(len(fac), n + 1):
            fac.append((fac[-1]*i) % mod)
            inv.append((-inv[mod % i]*(mod//i)) % mod)
            finv.append((finv[-1]*inv[-1]) % mod)
    
    def __call__(self, n, r):
        if not 0 <= r <= n:
            return 0
        self._make(max(n, r))
        fac, finv = self._fac, self._finv
        return fac[n]*finv[r]*finv[n-r] % self.mod

def main():
    N, K = map(int, input().split())
    P = [int(x) for x in input().split()]
    calc = HistogramCalculator(N)
    ans = calc.calculate_expected_inversions(P, K)
    print(ans)

if __name__ == "__main__":
    main()