def count_valid_pairs(a):
    from collections import Counter
    freq = Counter(a)
    total = 0
    for value in freq.values():
        total += (value * (value - 1)) // 2
    return total

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    results = []
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        result = count_valid_pairs(a)
        results.append(str(result))
    print('
'.join(results))

if __name__ == "__main__":
    main()