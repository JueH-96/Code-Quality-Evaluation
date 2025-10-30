import sys

def main():
    max_val = 10**5
    # Sieve to compute smallest prime factors (SPF)
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute bigomega for each number
    bigomega = [0] * (max_val + 1)
    for n in range(2, max_val + 1):
        x = n
        cnt = 0
        while x != 1:
            p = spf[x]
            while x % p == 0:
                cnt += 1
                x //= p
        bigomega[n] = cnt
    
    # Read input and compute XOR sum
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:N+1]))
    xor_sum = 0
    for a in A:
        xor_sum ^= bigomega[a]
    
    print("Anna" if xor_sum != 0 else "Bruno")

if __name__ == '__main__':
    main()