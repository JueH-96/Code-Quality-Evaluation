import sys

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    D = list(map(int, sys.stdin.readline().split()))
    
    # Sort coupons by descending D, then descending L
    coupons = sorted(zip(L, D), key=lambda x: (-x[1], -x[0]))
    # Sort items in descending order
    P.sort(reverse=True)
    
    parent = list(range(N + 2))  # To handle N-1 +1 safely
    total_saving = 0
    
    for l, d in coupons:
        # Binary search for the rightmost index with P >= l
        low, high, right = 0, N-1, -1
        while low <= high:
            mid = (low + high) // 2
            if P[mid] >= l:
                right = mid
                low = mid + 1
            else:
                high = mid - 1
        if right == -1:
            continue
        
        # Find the earliest available item
        available = find(parent, 0)
        if available <= right:
            total_saving += d
            # Union the item with the next
            parent[available] = available + 1
    
    print(sum(P) - total_saving)

if __name__ == "__main__":
    main()