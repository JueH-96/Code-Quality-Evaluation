import sys

# ---------- sieve that stores the smallest prime factor ----------
def build_spf(limit: int):
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:                       # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))

    maxA = max(A)
    spf = build_spf(maxA)

    xor_sum = 0
    for a in A:
        cnt = 0
        while a > 1:
            p = spf[a]
            while a % p == 0:
                a //= p
                cnt += 1          # add multiplicity
        xor_sum ^= cnt

    print("Anna" if xor_sum else "Bruno")


if __name__ == "__main__":
    main()