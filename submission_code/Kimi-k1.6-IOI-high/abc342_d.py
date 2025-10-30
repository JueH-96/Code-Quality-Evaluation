import sys
from collections import defaultdict

def main():
    max_a = 200000
    spf = list(range(max_a + 1))
    for i in range(2, int(max_a**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_a + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_square_free(x):
        if x == 0:
            return 0
        res = 1
        while x != 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                cnt += 1
                x = x // p
            if cnt % 2 == 1:
                res *= p
        return res

    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    k = A.count(0)
    non_zeros = [x for x in A if x != 0]
    zero_pairs = k * (k - 1) // 2 + k * len(non_zeros)
    
    freq = defaultdict(int)
    for x in non_zeros:
        s = get_square_free(x)
        freq[s] += 1
    
    non_zero_pairs = 0
    for c in freq.values():
        non_zero_pairs += c * (c - 1) // 2
    
    total = zero_pairs + non_zero_pairs
    print(total)

if __name__ == "__main__":
    main()