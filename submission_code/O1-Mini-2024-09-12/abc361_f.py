import sys

def integer_nth_root(n, k):
    low = 1
    high = n
    while low <= high:
        mid = (low + high) // 2
        p = pow(mid, k)
        if p == n:
            return mid
        elif p < n:
            low = mid +1
        else:
            high = mid -1
    return high

def main():
    N = int(sys.stdin.read())
    s = set()
    s.add(1)
    max_b = 60
    for b in range(2, max_b+1):
        a_max = integer_nth_root(N, b)
        if a_max < 2:
            continue
        for a in range(2, a_max+1):
            p = pow(a, b)
            s.add(p)
    print(len(s))

if __name__ == "__main__":
    main()