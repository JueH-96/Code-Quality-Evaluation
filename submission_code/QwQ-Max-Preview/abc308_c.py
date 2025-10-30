import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for idx in range(1, n + 1):
        a, b = map(int, sys.stdin.readline().split())
        people.append((a, a + b, idx))
    
    def compare(x, y):
        a_i, t_i, idx_i = x
        a_j, t_j, idx_j = y
        left = a_i * t_j
        right = a_j * t_i
        if left > right:
            return -1
        elif left < right:
            return 1
        else:
            return idx_i - idx_j
    
    people.sort(key=cmp_to_key(compare))
    result = [str(p[2]) for p in people]
    print(' '.join(result))

if __name__ == "__main__":
    main()