import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A_N = A[-1]
    max_i = (1 << 100) - A_N - 1

    candidates = set()

    # Generate all possible i candidates
    for m in range(1, 61):
        power = 1 << m
        for a in A:
            i_candidate = (power - 1) - a
            if i_candidate >= 1:
                candidates.add(i_candidate)

    def is_mountain(x):
        flips = 0
        while True:
            if x == 0:
                return False  # x starts from 1
            m = 1 << (x.bit_length() - 1)
            if x == m:
                return flips % 2 == 1
            x = 2 * m - x
            flips += 1

    max_sum = 0
    for i in candidates:
        if i > max_i:
            continue
        cnt = 0
        for a in A:
            x = i + a
            if is_mountain(x):
                cnt += 1
        if cnt > max_sum:
            max_sum = cnt

    print(max_sum)

if __name__ == '__main__':
    main()