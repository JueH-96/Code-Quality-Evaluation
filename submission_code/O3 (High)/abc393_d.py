import sys

def main() -> None:
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    s = data[1].strip()

    ones_pos = []
    for i, ch in enumerate(s):
        if ch == '1':
            ones_pos.append(i)         # 0-based positions of every '1'

    k = len(ones_pos)                  # number of 1s (k ≥ 1 as guaranteed)

    # a_i = p_i - i   (p_i : position of i-th 1, i starts from 0)
    shifted = [ones_pos[i] - i for i in range(k)]

    median = shifted[k // 2]           # any median minimises the sum of |a_i - m|

    moves = sum(abs(x - median) for x in shifted)
    print(moves)

if __name__ == "__main__":
    main()