import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for i in range(1, n+1):
        a, b = map(int, sys.stdin.readline().split())
        people.append((a, b, i))
    
    def compare(x, y):
        lhs = x[0] * y[1]
        rhs = y[0] * x[1]
        if lhs > rhs:
            return -1
        elif lhs < rhs:
            return 1
        else:
            return x[2] - y[2]
    
    people_sorted = sorted(people, key=cmp_to_key(compare))
    result = [str(p[2]) for p in people_sorted]
    print(' '.join(result))

if __name__ == "__main__":
    main()