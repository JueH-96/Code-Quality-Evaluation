import sys
from collections import defaultdict

def main():
    max_num = 200000
    spf = list(range(max_num + 1))
    for i in range(2, int(max_num**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_num+1, i):
                if spf[j] == j:
                    spf[j] = i
    
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    zero_count = 0
    sum_non_zero_pairs = 0
    sum_zero_non_zero = 0
    square_free_counts = defaultdict(int)
    
    for num in A:
        if num == 0:
            zero_count += 1
        else:
            x = num
            s = 1
            while x != 1:
                p = spf[x]
                count = 0
                while x % p == 0:
                    count += 1
                    x = x // p
                if count % 2 != 0:
                    s *= p
            sum_non_zero_pairs += square_free_counts[s]
            square_free_counts[s] += 1
            sum_zero_non_zero += zero_count
    
    total = sum_non_zero_pairs + (zero_count * (zero_count - 1)) // 2 + sum_zero_non_zero
    print(total)

if __name__ == '__main__':
    main()