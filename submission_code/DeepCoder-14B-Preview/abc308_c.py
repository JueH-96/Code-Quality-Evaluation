import sys
from functools import cmp_to_key

def main():
    n = int(sys.stdin.readline())
    people = []
    for i in range(n):
        a, b = map(int, sys.stdin.readline().split())
        people.append((a, b, i + 1))  # (A_i, B_i, index)
    
    def compare(x, y):
        a1, b1, idx1 = x
        a2, b2, idx2 = y
        total1 = a1 + b1
        total2 = a2 + b2
        left = a1 * total2
        right = a2 * total1
        if left > right:
            return -1  # x should come before y
        elif left < right:
            return 1   # y should come before x
        else:
            # Same success rate, compare indices
            if idx1 < idx2:
                return -1
            else:
                return 1
    
    # Sort the people using the custom comparator
    people_sorted = sorted(people, key=cmp_to_key(compare))
    
    # Extract the indices in the sorted order
    result = [str(p[2]) for p in people_sorted]
    print(' '.join(result))

if __name__ == "__main__":
    main()