import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    # Precompute smallest prime factors (SPF) up to 200000
    max_spf = 200000
    spf = list(range(max_spf + 1))
    for i in range(2, int(max_spf**0.5) + 1):
        if spf[i] == i:  # i is a prime
            for j in range(i*i, max_spf + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Calculate square-free parts for non-zero elements
    freq = defaultdict(int)
    for x in a:
        if x == 0:
            continue
        current = x
        square_free = 1
        while current > 1:
            p = spf[current]
            cnt = 0
            while current % p == 0:
                cnt += 1
                current = current // p
            if cnt % 2 == 1:
                square_free *= p
        freq[square_free] += 1
    
    # Calculate zero pairs
    cnt_zero = a.count(0)
    non_zero_count = n - cnt_zero
    zero_pairs = n * (n - 1) // 2 - non_zero_count * (non_zero_count - 1) // 2
    
    # Calculate non-zero pairs
    non_zero_pairs = 0
    for c in freq.values():
        non_zero_pairs += c * (c - 1) // 2
    
    total = zero_pairs + non_zero_pairs
    print(total)

if __name__ == "__main__":
    main()