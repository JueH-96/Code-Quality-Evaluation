import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for idx in range(n):
        a, b = map(int, sys.stdin.readline().split())
        people.append((idx + 1, a, b))
    
    def compare(p1, p2):
        a1, b1 = p1[1], p1[2]
        a2, b2 = p2[1], p2[2]
        left = a1 * (a2 + b2)
        right = a2 * (a1 + b1)
        if left > right:
            return -1
        elif left < right:
            return 1
        else:
            return -1 if p1[0] < p2[0] else 1
    
    people_sorted = sorted(people, key=cmp_to_key(compare))
    print(' '.join(map(str, [p[0] for p in people_sorted])))

if __name__ == "__main__":
    main()