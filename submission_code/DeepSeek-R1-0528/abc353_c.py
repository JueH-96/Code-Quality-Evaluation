MOD_VAL = 10**8

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    total_sum = sum(arr)
    T = total_sum * (n - 1)
    
    arr.sort()
    left = 0
    right = n - 1
    count = 0
    while left < right:
        if arr[left] + arr[right] >= MOD_VAL:
            count += right - left
            right -= 1
        else:
            left += 1
            
    result = T - MOD_VAL * count
    print(result)

if __name__ == "__main__":
    main()