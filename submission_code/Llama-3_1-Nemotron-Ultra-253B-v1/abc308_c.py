import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for idx in range(1, n + 1):
        a, b = map(int, sys.stdin.readline().split())
        people.append((a, b, idx))
    
    def compare(x, y):
        a, b, ix = x
        c, d, iy = y
        left = a * (c + d)
        right = c * (a + b)
        if left > right:
            return -1
        elif left < right:
            return 1
        else:
            return -1 if ix < iy else 1
    
    sorted_people = sorted(people, key=cmp_to_key(compare))
    print(' '.join(str(p[2]) for p in sorted_people))

if __name__ == "__main__":
    main()