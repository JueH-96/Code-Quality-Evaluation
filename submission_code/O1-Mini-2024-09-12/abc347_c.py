# YOUR CODE HERE
import sys

def readints():
    return list(map(int, sys.stdin.read().split()))

def intersect_intervals(list1, list2):
    i, j = 0, 0
    result = []
    while i < len(list1) and j < len(list2):
        a, b = list1[i]
        c, d = list2[j]
        # Find overlap
        start = max(a, c)
        end = min(b, d)
        if start <= end:
            result.append((start, end))
        if b < d:
            i += 1
        else:
            j += 1
    return result

def main():
    data = readints()
    N, A, B = data[:3]
    D = data[3:]
    M = A + B
    possible = [(0, M-1)]
    for d in D:
        rem = d % M
        L = (M - rem) % M
        R = (L + A -1) % M
        if R >= L:
            new_range = [(L, R)]
        else:
            new_range = [(0, R), (L, M-1)]
        possible = intersect_intervals(possible, new_range)
        if not possible:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()