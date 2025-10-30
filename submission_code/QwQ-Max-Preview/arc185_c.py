import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    x = int(input[1])
    a = list(map(int, input[2:2+n]))
    
    value_indices = {}
    for idx, num in enumerate(a):
        if num not in value_indices:
            value_indices[num] = []
        value_indices[num].append(idx)
    
    for i in range(n):
        for j in range(i+1, n):
            required = x - a[i] - a[j]
            if required not in value_indices:
                continue
            indices = value_indices[required]
            pos = bisect.bisect_right(indices, j)
            if pos < len(indices):
                k = indices[pos]
                print(f"{i+1} {j+1} {k+1}")
                return
    print(-1)

if __name__ == "__main__":
    main()