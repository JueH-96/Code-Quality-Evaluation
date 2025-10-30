def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))

    # Precompute smallest prime factors up to 2*10^5
    MAX_A = 2 * 10**5
    SPF = [0] * (MAX_A + 1)
    for i in range(2, MAX_A + 1):
        if SPF[i] == 0:
            SPF[i] = i
            for j in range(i*i, MAX_A + 1, i):
                if SPF[j] == 0:
                    SPF[j] = i

    # Function to compute square-free part
    def square_free(n):
        if n == 0:
            return 0
        res = 1
        while n > 1:
            p = SPF[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count % 2 == 1:
                res *= p
        return res

    # Count zeros
    Z = A.count(0)

    # Compute square-free parts for non-zero elements
    freq = defaultdict(int)
    for a in A:
        if a == 0:
            continue
        sf = square_free(a)
        freq[sf] += 1

    # Calculate pairs for non-zero groups
    total = 0
    for count in freq.values():
        total += count * (count - 1) // 2

    # Calculate zero-involving pairs
    if Z >= 1:
        total += Z * (Z - 1) // 2 + Z * (N - Z)

    print(total)

if __name__ == "__main__":
    main()