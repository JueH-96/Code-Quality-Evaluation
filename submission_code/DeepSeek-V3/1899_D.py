import sys
from collections import defaultdict

def count_valid_pairs(a):
    freq = defaultdict(int)
    for num in a:
        freq[num] += 1
    count = 0
    # Count pairs where a_i == a_j
    for key in freq:
        cnt = freq[key]
        if cnt >= 2:
            count += cnt * (cnt - 1) // 2
    # Count pairs where a_i == 1 and a_j == 2
    cnt1 = freq.get(1, 0)
    cnt2 = freq.get(2, 0)
    count += cnt1 * cnt2
    return count

def main():
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
        print(count_valid_pairs(a))

if __name__ == "__main__":
    main()