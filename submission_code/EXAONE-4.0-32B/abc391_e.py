def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip().replace(' ', '')
    
    arr = []
    for char in s:
        if char == '0':
            arr.append((0, 1))
        else:
            arr.append((1, 0))
            
    total_levels = n
    for level in range(total_levels):
        new_arr = []
        for i in range(0, len(arr), 3):
            a0, a1 = arr[i]
            b0, b1 = arr[i+1]
            c0, c1 = arr[i+2]
            
            cost0 = min(a0 + b0 + c0, 
                       a0 + b0 + c1,
                       a0 + b1 + c0,
                       a1 + b0 + c0)
            cost1 = min(a1 + b1 + c1,
                       a1 + b1 + c0,
                       a1 + b0 + c1,
                       a0 + b1 + c1)
            new_arr.append((cost0, cost1))
        arr = new_arr
        
    root = arr[0]
    if root[0] == 0:
        print(root[1])
    else:
        print(root[0])

if __name__ == "__main__":
    main()