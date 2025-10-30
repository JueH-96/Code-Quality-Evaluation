import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    x = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    value_indices = defaultdict(list)
    for idx, num in enumerate(a):
        value_indices[num].append(idx)
    
    for i in range(n):
        a_i = a[i]
        for j in range(i + 1, n):
            a_j = a[j]
            required = x - a_i - a_j
            if required not in value_indices:
                continue
            indices_list = value_indices[required]
            k_pos = bisect.bisect_right(indices_list, j)
            if k_pos < len(indices_list):
                k = indices_list[k_pos]
                print(i + 1, j + 1, k + 1)
                return
    print(-1)

if __name__ == "__main__":
    main()