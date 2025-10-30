import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    x = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    index_map = defaultdict(list)
    for idx, val in enumerate(a):
        index_map[val].append(idx)
    
    for i in range(n):
        for j in range(i + 1, n):
            c = x - a[i] - a[j]
            if c < 0:
                continue
            if c not in index_map:
                continue
            indices = index_map[c]
            # Find the first index in indices that is > j
            pos = bisect.bisect_right(indices, j)
            if pos < len(indices):
                k = indices[pos]
                if k > j:
                    print(i + 1, j + 1, k + 1)
                    return
    print(-1)

if __name__ == "__main__":
    main()