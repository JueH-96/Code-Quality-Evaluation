import sys
from functools import cmp_to_key

def main():
    input = sys.stdin.readline
    n = int(input())
    people = []
    for i in range(1, n+1):
        parts = input().split()
        # in case of blank lines
        while not parts:
            parts = input().split()
        a = int(parts[0])
        b = int(parts[1])
        people.append((a, a + b, i))

    def compare(x, y):
        # Compare success rates: x[0]/x[1] vs y[0]/y[1]
        lhs = x[0] * y[1]
        rhs = y[0] * x[1]
        if lhs > rhs:
            return -1   # x has higher rate => comes first
        if lhs < rhs:
            return 1    # y has higher rate => x comes after
        # tie => ascending person number
        return x[2] - y[2]

    people.sort(key=cmp_to_key(compare))
    result = [str(p[2]) for p in people]
    sys.stdout.write(" ".join(result))

if __name__ == "__main__":
    main()