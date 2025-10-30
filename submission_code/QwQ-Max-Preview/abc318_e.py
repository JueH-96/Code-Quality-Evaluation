import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    groups = defaultdict(list)
    for idx in range(n):
        num = a[idx]
        groups[num].append(idx + 1)  # 1-based index

    total = 0
    for x in groups:
        pos = groups[x]
        m = len(pos)
        if m < 2:
            continue
        sum1 = 0
        sum2 = 0
        for b in range(m):
            p_b = pos[b]
            sum1 += (p_b - b) * b
        for a in range(m):
            p_a = pos[a]
            sum2 += (p_a - a) * ((m - 1) - a)
        total += (sum1 - sum2)
    print(total)

if __name__ == "__main__":
    main()