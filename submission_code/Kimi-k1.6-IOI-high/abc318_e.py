from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    a = list(map(int, data[1:n+1]))
    
    pos = defaultdict(list)
    for idx in range(n):
        num = a[idx]
        pos[num].append(idx + 1)  # 1-based index
    
    result = 0
    for x in pos:
        lst = pos[x]
        m = len(lst)
        if m < 2:
            continue
        total = 0
        for i in range(m):
            p = lst[i]
            term = p - (i + 1)
            coeff = (2 * (i + 1) - m - 1)
            total += term * coeff
        result += total
    print(result)

if __name__ == "__main__":
    main()