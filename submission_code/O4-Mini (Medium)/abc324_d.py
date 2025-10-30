import sys
import threading

def main():
    import sys, math
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()
    # frequency of digits in S
    freq_S = [0]*10
    for ch in S:
        freq_S[ord(ch)-48] += 1
    t0 = freq_S[0]
    # target counts for digits 1..9
    target = freq_S[1:]
    # sum of digits mod 9 for filter
    ssd = 0
    for ch in S:
        ssd += (ord(ch)-48)
    ssd %= 9

    # maximum k so that k*k < 10^N
    # i.e. k <= isqrt(10^N - 1)
    X = 10**N
    k_max = math.isqrt(X - 1)

    ans = 0
    freq_sq = [0]*10
    r = 0  # r = k % 9

    for k in range(k_max+1):
        # filter by square mod 9 == sum digits mod9
        if (r*r) % 9 == ssd:
            sq = k*k
            s_str = str(sq)
            # count digits of sq into freq_sq, track seen digits
            cz = []
            for ch in s_str:
                d = ord(ch) - 48
                freq_sq[d] += 1
                cz.append(d)
            # check match: non-zero digit counts match exactly, zero count <= t0
            if freq_sq[0] <= t0 and freq_sq[1:] == target:
                ans += 1
            # reset freq_sq for next iteration
            for d in cz:
                freq_sq[d] = 0
        # update r = (k+1) % 9
        r += 1
        if r == 9:
            r = 0

    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()