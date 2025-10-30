import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        B = [i for i, c in enumerate(s) if c == 'B']
        if not B:
            print(0)
            continue
        count = 0
        last_end = -1
        for pos in B:
            if pos >= last_end:
                s_win = min(pos, n - k)
                last_end = s_win + k
                count += 1
        print(count)

if __name__ == "__main__":
    main()