import sys
import functools

def compare(x, y):
    a_x, b_x, idx_x = x
    a_y, b_y, idx_y = y
    left = a_x * (a_y + b_y)
    right = a_y * (a_x + b_x)
    if left > right:
        return -1
    elif left < right:
        return 1
    else:
        return idx_x - idx_y

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    ptr = 1
    people = []
    for i in range(n):
        a = int(input[ptr])
        b = int(input[ptr + 1])
        ptr += 2
        people.append((a, b, i + 1))
    people_sorted = sorted(people, key=functools.cmp_to_key(compare))
    print(' '.join(str(p[2]) for p in people_sorted))

if __name__ == "__main__":
    main()