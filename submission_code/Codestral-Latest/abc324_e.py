import sys
from collections import defaultdict

def is_subsequence(s, t):
    it = iter(s)
    return all(c in it for c in t)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    T = data[1]
    S = data[2:]

    count = 0

    for i in range(N):
        for j in range(N):
            if is_subsequence(S[i] + S[j], T):
                count += 1

    print(count)

if __name__ == "__main__":
    main()