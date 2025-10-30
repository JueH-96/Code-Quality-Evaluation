import sys

def main():
    input_data = sys.stdin.readline().split()
    if not input_data:
        return
    N = int(input_data[0])
    L = int(input_data[1])
    R = int(input_data[2])

    ans = 0
    cur = L
    # Cover [L, R] by dyadic (power-of-two) aligned segments
    while cur <= R:
        remain = R - cur + 1
        # maximum i allowed by alignment of cur
        if cur == 0:
            tz = N
        else:
            # number of trailing zeros in cur
            tz = (cur & -cur).bit_length() - 1
        # maximum i allowed by length
        i_len = remain.bit_length() - 1
        i = min(tz, i_len)
        j = cur >> i

        # ask the sum of the block [2^i * j, 2^i * (j+1) - 1]
        print("?", i, j, flush=True)
        resp = sys.stdin.readline().strip()
        if not resp:
            return
        t = int(resp)
        if t < 0:
            # invalid query or judge error
            return
        ans = (ans + t) % 100
        cur += (1 << i)

    # output the final answer
    print("!", ans, flush=True)

if __name__ == "__main__":
    main()