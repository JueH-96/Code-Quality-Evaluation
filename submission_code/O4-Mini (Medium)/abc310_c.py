import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    idx = 1
    seen = set()
    for _ in range(n):
        s = data[idx]
        idx += 1
        rs = s[::-1]
        # take the lexicographically smaller of s and its reverse
        if rs < s:
            s = rs
        seen.add(s)
    print(len(seen))

if __name__ == "__main__":
    main()