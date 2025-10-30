import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for idx in range(n):
        a, b = map(int, sys.stdin.readline().split())
        people.append((a, b, idx + 1))
    
    def compare(x, y):
        xa, xb, x_idx = x
        ya, yb, y_idx = y
        left = xa * (ya + yb)
        right = ya * (xa + xb)
        if left > right:
            return -1
        elif left < right:
            return 1
        else:
            return -1 if x_idx < y_idx else 1
    
    sorted_people = sorted(people, key=cmp_to_key(compare))
    print(' '.join(str(p[2]) for p in sorted_people))

if __name__ == "__main__":
    main()