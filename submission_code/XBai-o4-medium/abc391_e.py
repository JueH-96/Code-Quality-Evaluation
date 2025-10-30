import sys
from functools import lru_cache

def main():
    N = int(sys.stdin.readline())
    A = sys.stdin.readline().strip()

    def compute_current():
        current = list(A)
        for _ in range(N):
            new_current = []
            third = len(current) // 3
            for i in range(third):
                triple = current[i*3 : (i+1)*3]
                cnt0 = triple.count('0')
                cnt1 = 3 - cnt0
                if cnt0 > cnt1:
                    new_current.append('0')
                else:
                    new_current.append('1')
            current = new_current
        return current[0]

    current_val = compute_current()
    target_val = 1 - int(current_val)

    @lru_cache(maxsize=None)
    def min_cost(start, m, target):
        if m == 0:
            return 0 if int(A[start]) == target else 1
        else:
            part_len = 3 ** (m-1)
            part0 = start
            part1 = start + part_len
            part2 = start + 2 * part_len
            min_total = float('inf')
            for a in [0, 1]:
                for b in [0, 1]:
                    for c in [0, 1]:
                        cnt0 = [a, b, c].count(0)
                        cnt1 = 3 - cnt0
                        if (target == 0 and cnt0 >= 2) or (target == 1 and cnt1 >= 2):
                            cost = min_cost(part0, m-1, a) + min_cost(part1, m-1, b) + min_cost(part2, m-1, c)
                            if cost < min_total:
                                min_total = cost
            return min_total

    result = min_cost(0, N, target_val)
    print(result)

if __name__ == '__main__':
    main()