import sys
from collections import defaultdict

def is_subsequence(s, t):
    it = iter(t)
    return all(char in it for char in s)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    T = data[1]
    strings = data[2:]

    count = 0
    n = len(strings)

    for i in range(n):
        for j in range(n):
            if is_subsequence(T, strings[i] + strings[j]):
                count += 1

    print(count)

if __name__ == "__main__":
    main()