import sys

def main() -> None:
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()

    ans = 0          # final answer
    t  = 0           # current value of (10^{len+1}-1)//9 for the suffix length

    # iterate from rightmost digit to leftmost
    for idx in range(N - 1, -1, -1):         # idx : 0-based index
        if idx == N - 1:
            t = 1                            # (10^1 - 1)/9  for len = 0
        else:
            t = t * 10 + 1                   #  t = 10 * t_{next} + 1

        k = idx + 1                          # position (1-based)
        digit = ord(S[idx]) - 48             # convert char to int (faster than int())

        ans += digit * t * k                 # add contribution of this digit

    print(ans)

if __name__ == "__main__":
    main()