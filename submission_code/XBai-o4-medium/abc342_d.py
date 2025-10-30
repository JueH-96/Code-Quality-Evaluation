import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    max_A = 200000
    max_val = max_A + 1
    
    # Precompute smallest prime factors (spf)
    spf = list(range(max_val))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is a prime
            for j in range(i*i, max_val, i):
                if spf[j] == j:
                    spf[j] = i
    
    count_zero = 0
    freq = defaultdict(int)
    
    for num in A:
        if num == 0:
            count_zero += 1
        else:
            current = num
            square_free = 1
            while current > 1:
                p = spf[current]
                count_p = 0
                while current % p == 0:
                    count_p += 1
                    current //= p
                if count_p % 2 == 1:
                    square_free *= p
            freq[square_free] += 1
    
    ans = (count_zero * (count_zero - 1)) // 2 + count_zero * (N - count_zero)
    
    for key in freq:
        m = freq[key]
        ans += (m * (m - 1)) // 2
    
    print(ans)

if __name__ == "__main__":
    main()