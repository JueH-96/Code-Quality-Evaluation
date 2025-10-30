def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    total_subarrays = n * (n + 1) // 2
    
    pos_dict = {}
    for idx, num in enumerate(A):
        if num not in pos_dict:
            pos_dict[num] = []
        pos_dict[num].append(idx)
    
    ans = 0
    for positions in pos_dict.values():
        gaps = []
        gaps.append(positions[0])
        
        for i in range(1, len(positions)):
            gap_length = positions[i] - positions[i-1] - 1
            gaps.append(gap_length)
            
        gaps.append(n - 1 - positions[-1])
        
        without_x = 0
        for g in gaps:
            without_x += g * (g + 1) // 2
            
        ans += (total_subarrays - without_x)
        
    print(ans)

if __name__ == "__main__":
    main()