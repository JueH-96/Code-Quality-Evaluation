import bisect

def main():
    n = int(input().strip())
    L = []
    R = []
    for _ in range(n):
        l, r = map(int, input().split())
        L.append(l)
        R.append(r)
    
    total_pairs = n * (n - 1) // 2
    L.sort()
    disjoint = 0
    for b in R:
        idx = bisect.bisect_right(L, b)
        disjoint += (n - idx)
        
    print(total_pairs - disjoint)

if __name__ == "__main__":
    main()