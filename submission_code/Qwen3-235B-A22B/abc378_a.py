from collections import Counter

def main():
    import sys
    input = sys.stdin.read().split()
    a = list(map(int, input))
    counts = Counter(a)
    total = sum(v // 2 for v in counts.values())
    print(total)

if __name__ == "__main__":
    main()