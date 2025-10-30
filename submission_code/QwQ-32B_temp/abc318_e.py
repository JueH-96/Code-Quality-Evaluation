import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))

    positions = defaultdict(list)
    for idx in range(N):
        num = A[idx]
        pos = idx + 1  # 1-based position
        positions[num].append(pos)

    ans = 0
    for plist in positions.values():
        m = len(plist)
        if m < 2:
            continue
        S1 = 0
        sum_terms = 0
        for j in range(m):
            term_j = plist[j] - j
            S1 += term_j * j
            sum_terms += term_j
        contribution = 2 * S1 - (m - 1) * sum_terms
        ans += contribution
    print(ans)

if __name__ == "__main__":
    main()