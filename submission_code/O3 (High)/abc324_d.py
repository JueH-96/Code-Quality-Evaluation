import sys
import math

def digit_count_with_padding(x: int, N: int):
    """
    Return a tuple with the amount of each decimal digit (0..9) occurring
    in the representation of x, padded with leading zeroes so that the
    total length equals N.
    """
    cnt = [0] * 10
    if x == 0:
        cnt[0] = 1
        length = 1
    else:
        length = 0
        while x:
            x, d = divmod(x, 10)
            cnt[d] += 1
            length += 1
    cnt[0] += N - length   # add leading zeroes
    return tuple(cnt)

def main() -> None:
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # 1. multiset of digits of S
    target = [0] * 10
    for ch in S:
        target[ord(ch) - 48] += 1
    target = tuple(target)

    # 2. upper bound for the square root
    max_k = math.isqrt(10 ** N - 1)

    # 3. enumerate squares
    squares_found = set()
    for k in range(max_k + 1):
        sq = k * k
        if digit_count_with_padding(sq, N) == target:
            squares_found.add(sq)

    # 4. answer
    print(len(squares_found))

if __name__ == "__main__":
    main()