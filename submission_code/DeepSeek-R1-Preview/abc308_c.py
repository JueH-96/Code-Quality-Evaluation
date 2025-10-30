import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for idx in range(n):
        a, b = map(int, sys.stdin.readline().split())
        people.append((a, b, idx + 1))
    
    def compare(p1, p2):
        a1, b1, i1 = p1
        a2, b2, i2 = p2
        left = a1 * b2
        right = a2 * b1
        if left > right:
            return -1
        elif left < right:
            return 1
        else:
            return -1 if i1 < i2 else 1
    
    people_sorted = sorted(people, key=cmp_to_key(compare))
    result = [p[2] for p in people_sorted]
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()