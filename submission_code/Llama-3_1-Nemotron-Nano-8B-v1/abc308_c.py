import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for i in range(n):
        a, b = map(int, sys.stdin.readline().split())
        people.append((a, b, i + 1))
    
    def compare(x, y):
        a1, b1, idx1 = x
        a2, b2, idx2 = y
        cross_x = a1 * (a2 + b2)
        cross_y = a2 * (a1 + b1)
        if cross_x > cross_y:
            return -1
        elif cross_x < cross_y:
            return 1
        else:
            return idx1 - idx2
    
    people.sort(key=cmp_to_key(compare))
    print(' '.join(str(p[2]) for p in people))

if __name__ == "__main__":
    main()