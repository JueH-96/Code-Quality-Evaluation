import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for i in range(n):
        a, b = map(int, sys.stdin.readline().split())
        s = a + b
        people.append((a, s, i + 1))  # Store A, s, and index

    def compare(x, y):
        a_A, a_S, a_i = x
        b_A, b_S, b_i = y
        left = a_A * b_S
        right = b_A * a_S
        if left > right:
            return -1
        elif left < right:
            return 1
        else:
            if a_i < b_i:
                return -1
            else:
                return 1

    people_sorted = sorted(people, key=cmp_to_key(compare))
    result = [str(p[2]) for p in people_sorted]
    print(' '.join(result))

if __name__ == '__main__':
    main()