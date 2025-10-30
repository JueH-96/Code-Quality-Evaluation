def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute smallest prime factors (SPF) up to 2*10^5
    max_A = 2 * 10**5
    SPF = [0] * (max_A + 1)
    for i in range(2, max_A + 1):
        if SPF[i] == 0:
            SPF[i] = i
            for j in range(i * 2, max_A + 1, i):
                if SPF[j] == 0:
                    SPF[j] = i
    
    # Count zeros
    c0 = A.count(0)
    
    # Function to compute square-free part
    def square_free(x):
        if x == 0:
            return 0
        sf = 1
        while x > 1:
            p = SPF[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt % 2 == 1:
                sf *= p
        return sf
    
    # Compute frequency of square-free parts for non-zero elements
    freq = {}
    for a in A:
        if a != 0:
            sf = square_free(a)
            if sf in freq:
                freq[sf] += 1
            else:
                freq[sf] = 1
    
    # Calculate pairs for non-zero elements
    pairs_non_zero = 0
    for c in freq.values():
        pairs_non_zero += c * (c - 1) // 2
    
    # Calculate pairs involving at least one zero
    pairs_with_zero = c0 * (N - c0) + c0 * (c0 - 1) // 2
    
    # Total pairs
    total = pairs_with_zero + pairs_non_zero
    print(total)

if __name__ == "__main__":
    main()