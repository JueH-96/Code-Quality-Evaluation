import sys
from bisect import bisect_right
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    R = int(input[1])
    C = int(input[2])
    S = input[3]

    sum_dr = [0] * (N + 1)
    sum_dc = [0] * (N + 1)

    pos_map = defaultdict(list)
    pos_map[(0, 0)].append(0)

    for i in range(1, N + 1):
        c = S[i - 1]
        if c == 'N':
            dr, dc = -1, 0
        elif c == 'S':
            dr, dc = 1, 0
        elif c == 'E':
            dr, dc = 0, 1
        elif c == 'W':
            dr, dc = 0, -1
        else:
            dr, dc = 0, 0  # should not happen

        sum_dr[i] = sum_dr[i - 1] + dr
        sum_dc[i] = sum_dc[i - 1] + dc

        pos_map[(sum_dr[i], sum_dc[i])].append(i)

    res = []
    for t in range(1, N + 1):
        target_dr = sum_dr[t] - R
        target_dc = sum_dc[t] - C
        key = (target_dr, target_dc)

        if key not in pos_map:
            res.append('0')
            continue

        k_list = pos_map[key]
        idx = bisect_right(k_list, t)
        if idx > 0:
            res.append('1')
        else:
            res.append('0')

    print(''.join(res))

if __name__ == "__main__":
    main()