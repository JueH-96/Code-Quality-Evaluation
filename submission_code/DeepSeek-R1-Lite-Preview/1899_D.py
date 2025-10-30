from collections import Counter

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        count = Counter(a)
        total = 0
        for val in count:
            c = count[val]
            total += c * (c - 1) // 2
        total += count.get(1, 0) * count.get(2, 0)
        print(total)

if __name__ == "__main__":
    main()