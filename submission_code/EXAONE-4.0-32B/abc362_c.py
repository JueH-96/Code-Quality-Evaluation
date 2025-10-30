def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    n = int(data[0].strip())
    intervals = []
    min_total = 0
    max_total = 0
    index = 1
    for i in range(n):
        parts = data[index].split()
        index += 1
        l = int(parts[0])
        r = int(parts[1])
        intervals.append((l, r))
        min_total += l
        max_total += r
        
    if min_total > 0 or max_total < 0:
        print("No")
    else:
        D = -min_total
        res = []
        for (l, r) in intervals:
            add_val = min(D, r - l)
            num = l + add_val
            res.append(str(num))
            D -= add_val
        print("Yes")
        print(" ".join(res))

if __name__ == "__main__":
    main()