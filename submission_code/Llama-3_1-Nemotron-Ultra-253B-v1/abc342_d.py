import sys
from collections import defaultdict

def main():
    max_A = 200000
    spf = list(range(max_A + 1))
    for i in range(2, int(max_A**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_A+1, i):
                if spf[j] == j:
                    spf[j] = i

    def square_free(x):
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

    sum_zero_pairs = 0
    sum_non_zero_pairs = 0
    count_zeros = 0
    freq = defaultdict(int)

    for index in range(N):
        x = A[index]
        if x == 0:
            sum_zero_pairs += (N - index - 1)
            sum_non_zero_pairs += (index - count_zeros)
            count_zeros += 1
        else:
            s = square_free(x)
            freq[s] += 1

    ans = sum(m * (m - 1) // 2 for m in freq.values())
    ans += sum_zero_pairs + sum_non_zero_pairs
    print(ans)

if __name__ == "__main__":
    main()