import sys

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        freq = [0] * 26
        for c in s:
            freq[ord(c) - ord('a')] += 1
        O = sum(1 for x in freq if x % 2 != 0)
        m = sum(1 for x in freq if x > 0)
        max_s = min(m, k, 26)
        found = False
        for s_size in range(0, max_s + 1):
            if (k - s_size) >= 0 and (k - s_size) % 2 == 0:
                required_s_o = (O + s_size) // 2
                max_possible_s_o = min(O, s_size)
                if required_s_o <= max_possible_s_o:
                    found = True
                    break
        print("YES" if found else "NO")

if __name__ == "__main__":
    main()