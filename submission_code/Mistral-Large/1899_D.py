import sys
from collections import defaultdict

def count_pairs(a):
    count = 0
    freq = defaultdict(int)
    for num in a:
        freq[num] += 1

    for key in freq:
        count += freq[key] * (freq[key] - 1) // 2

    return count

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    t = int(data[index])
    index += 1
    results = []

    for _ in range(t):
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index + n]))
        index += n
        results.append(count_pairs(a))

    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == "__main__":
    main()