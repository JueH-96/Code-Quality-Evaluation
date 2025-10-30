def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    total_sub = n * (n + 1) // 2

    from collections import defaultdict
    pos_dict = defaultdict(list)
    for idx, num in enumerate(arr):
        pos_dict[num].append(idx)
    
    ans = 0
    for x in pos_dict:
        positions = pos_dict[x]
        without_x = 0
        gap_begin = positions[0]
        without_x += gap_begin * (gap_begin + 1) // 2
        
        for i in range(1, len(positions)):
            gap = positions[i] - positions[i-1] - 1
            without_x += gap * (gap + 1) // 2
            
        gap_end = n - 1 - positions[-1]
        without_x += gap_end * (gap_end + 1) // 2
        
        ans += total_sub - without_x
        
    print(ans)

if __name__ == '__main__':
    main()