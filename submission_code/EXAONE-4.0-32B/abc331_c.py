def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 0:
        print("")
        return
        
    max_val = max(A)
    arr = [0] * (max_val + 2)
    
    for num in A:
        arr[num] += num
            
    suf = [0] * (max_val + 2)
    for i in range(max_val, -1, -1):
        suf[i] = suf[i + 1] + arr[i]
        
    res_list = []
    for a in A:
        res_list.append(str(suf[a + 1]))
        
    print(" ".join(res_list))

if __name__ == "__main__":
    main()