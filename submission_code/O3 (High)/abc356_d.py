import sys

MOD = 998244353      # required modulus
MAX_BIT = 60         # because N , M  < 2**60

def ones_up_to(n: int, bit: int) -> int:
    """
    Number of integers k in [0, n] whose `bit`-th bit is 1
    (0-based, least–significant bit is bit 0)
    """
    period = 1 << (bit + 1)          # length of one 0/1 cycle for this bit
    full_cycles = (n + 1) // period  # complete cycles contained in 0…n
    rem = (n + 1) % period           # leftover part of the last incomplete cycle

    ones = full_cycles * (1 << bit)              # ones coming from complete cycles
    ones += max(0, rem - (1 << bit))             # ones coming from the leftover
    return ones


def main() -> None:
    N, M = map(int, sys.stdin.read().split())
    ans = 0

    for bit in range(MAX_BIT):              # check every bit that may appear
        if (M >> bit) & 1:                  # this bit of M must be 1, otherwise
                                            # k & M can't have it set
            cnt = ones_up_to(N, bit)        # how many k (0…N) have this bit = 1
            ans = (ans + cnt) % MOD         # add its contribution mod MOD

    print(ans % MOD)


if __name__ == "__main__":
    main()