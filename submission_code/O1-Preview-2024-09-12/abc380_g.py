# YOUR CODE HERE
import sys
import threading

MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    total_inv = 0

    # Compute total inversion count of P
    # We will use BIT (Fenwick Tree) to compute inversions
    class BIT:
        def __init__(self, n):
            self.size = n
            self.tree = [0]*(n+2)
        def update(self, idx, val):
            idx += 1
            while idx <= self.size+1:
                self.tree[idx] += val
                idx += idx & -idx
        def query(self, idx):
            idx += 1
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res

    max_val = max(P)

    bit = BIT(max_val+2)
    for i in range(N):
        # Since permutations are from 1 to N, and BIT is 1-indexed
        num = P[i]
        total_inv += i - bit.query(num)
        bit.update(num,1)

    # Now compute inversion counts for each window of size K
    # We can use the same approach for each window
    # Since K can be up to N, total time complexity is O(N log N)

    # We slide a window of size K over P and compute the inversion count in each window
    # For this, we can process the first window and then update inversion counts incrementally

    inv_in_windows = []
    bit_window = BIT(max_val+2)

    window = P[:K]
    inv_in_window = 0
    for i in range(K):
        num = window[i]
        inv_in_window += i - bit_window.query(num)
        bit_window.update(num,1)
    inv_in_windows.append(inv_in_window)

    for i in range(K,N):
        # Remove P[i-K] from BIT
        num_remove = P[i-K]
        bit_window.update(num_remove,-1)
        # Adjust inversion count when removing num_remove
        # Number of elements less than num_remove after it
        num_less_after_remove = bit_window.query(num_remove -1)  # All elements < num_remove
        num_inv_remove = num_less_after_remove  # Since it contributed inv when added

        inv_in_window -= num_inv_remove

        # Now, add P[i] to BIT
        num_add = P[i]
        num_inv_add = bit_window.query(max_val+1) - bit_window.query(num_add)
        inv_in_window += num_inv_add
        bit_window.update(num_add,1)
        inv_in_windows.append(inv_in_window)

    sum_inv_shuffled = sum(inv_in_windows)
    E_Inv_Shuffled = sum_inv_shuffled / (N - K +1)

    # Expected inversion count in random permutation of length K
    E_Inv_Random_Shuffled = K*(K-1)/4

    E_Inv_P = total_inv - E_Inv_Shuffled + E_Inv_Random_Shuffled

    from fractions import Fraction
    frac = Fraction(E_Inv_P).limit_denominator()
    P_num = frac.numerator
    Q_den = frac.denominator

    # Compute (P_num * Q_den^-1 mod MOD)
    def modinv(a, mod):
        return pow(a, mod - 2, mod)
    Q_inv = modinv(Q_den, MOD)
    ans = (P_num * Q_inv)%MOD
    print(ans)

threading.Thread(target=main).start()