import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    MAX = 10**5

    # Compute smallest prime factors (SPF)
    spf = [0] * (MAX + 1)
    for i in range(2, MAX + 1):
        if spf[i] == 0:
            spf[i] = i
            for j in range(i * i, MAX + 1, i):
                if spf[j] == 0:
                    spf[j] = i
        # For composite numbers, spf[i] is already set

    # Compute omega (number of prime factors with multiplicity)
    omega = [0] * (MAX + 1)
    for n in range(2, MAX + 1):
        x = n
        count = 0
        while x > 1:
            p = spf[x]
            while x % p == 0:
                count += 1
                x //= p
        omega[n] = count

    # Calculate XOR of all omega values for the given array
    xor_sum = 0
    for a in A:
        xor_sum ^= omega[a]

    print("Anna" if xor_sum != 0 else "Bruno")

if __name__ == "__main__":
    main()