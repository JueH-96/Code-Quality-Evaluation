def main():
    import sys
    from collections import defaultdict
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    groups = defaultdict(list)
    for index, num in enumerate(arr):
        groups[num].append(index)
        
    total_ans = 0
    for key, lst in groups.items():
        m = len(lst)
        if m < 2:
            continue
        s = 0
        for idx, pos in enumerate(lst):
            s += pos * (2 * idx - (m - 1))
        second_part = m * (m - 1) * (m + 1) // 6
        total_ans += s - second_part
        
    print(total_ans)

if __name__ == "__main__":
    main()