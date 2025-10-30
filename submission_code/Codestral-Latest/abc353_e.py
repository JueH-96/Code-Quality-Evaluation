import sys
from collections import defaultdict

def longest_common_prefix_length(s1, s2):
    min_length = min(len(s1), len(s2))
    for i in range(min_length):
        if s1[i] != s2[i]:
            return i
    return min_length

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    strings = data[1:]

    total_length = 0

    for i in range(N):
        for j in range(i + 1, N):
            total_length += longest_common_prefix_length(strings[i], strings[j])

    print(total_length)

if __name__ == "__main__":
    main()