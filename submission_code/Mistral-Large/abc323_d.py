import sys
from collections import defaultdict

def min_slimes():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    index = 1
    slime_counts = defaultdict(int)

    for _ in range(N):
        S = int(data[index])
        C = int(data[index + 1])
        slime_counts[S] += C
        index += 2

    total_slimes = sum(slime_counts.values())

    for size in sorted(slime_counts.keys()):
        count = slime_counts[size]
        while count > 1:
            new_size = size * 2
            if count % 2 == 0:
                slime_counts[new_size] += count // 2
                total_slimes -= count // 2
            else:
                slime_counts[new_size] += (count // 2) + 1
                total_slimes -= (count // 2)
            count = 0

    print(total_slimes)

if __name__ == "__main__":
    min_slimes()