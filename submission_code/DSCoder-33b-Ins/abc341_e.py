# YOUR CODE HERE

import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    s = list(map(int, sys.stdin.readline().strip()))
    flip = [0] * (n + 1)
    for _ in range(q):
        query = list(map(int, sys.stdin.readline().split()))
        if query[0] == 1:
            l, r = query[1:]
            flip[l - 1] += 1
            flip[r] -= 1
        else:
            l, r = query[1:]
            for i in range(l - 1, r):
                flip[i] += flip[i - 1]
            s_prime = [(s[i] + flip[i]) % 2 for i in range(l - 1, r)]
            if all(s_prime[i] != s_prime[i - 1] for i in range(1, r - l + 1)):
                print('Yes')
            else:
                print('No')

if __name__ == '__main__':
    main()