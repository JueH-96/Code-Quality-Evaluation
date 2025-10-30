import sys

def main():
    input = sys.stdin.readline
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        cnt = {}
        for x in a:
            cnt[x] = cnt.get(x, 0) + 1

        # Count pairs of equal a_i = a_j
        ans = 0
        for c in cnt.values():
            ans += c * (c - 1) // 2

        # Add pairs where (a_i, a_j) = (1, 2) or (2, 1)
        ans += cnt.get(1, 0) * cnt.get(2, 0)

        print(ans)

if __name__ == "__main__":
    main()