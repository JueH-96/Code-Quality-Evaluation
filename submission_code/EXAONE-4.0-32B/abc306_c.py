def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+3*n]))
    
    occ = [[] for _ in range(n+1)]
    
    for idx, num in enumerate(arr):
        occ[num].append(idx)
    
    res_list = []
    for i in range(1, n+1):
        mid_index = occ[i][1]
        res_list.append((mid_index, i))
        
    res_list.sort(key=lambda x: x[0])
    
    ans = [str(num) for _, num in res_list]
    print(" ".join(ans))

if __name__ == "__main__":
    main()