import math
import sys

def main():
    N = int(sys.stdin.readline().strip())
    ans = 0
    power = 2
    while power <= N:
        T = N // power
        max_m = math.isqrt(T)
        ans += (max_m + 1) // 2
        power <<= 1
    print(ans)

if __name__ == "__main__":
    main()