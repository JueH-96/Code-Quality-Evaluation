import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return
        
    n = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    if n < 3:
        print(-1)
        return
        
    arr_with_index = [(A[i], i) for i in range(n)]
    arr_with_index.sort(key=lambda x: x[0])
    
    found = False
    for i in range(n-2):
        a_val, a_idx = arr_with_index[i]
        if i > 0 and a_val == arr_with_index[i-1][0]:
            continue
            
        if a_val + arr_with_index[i+1][0] + arr_with_index[i+2][0] > X:
            break
            
        if a_val + arr_with_index[-1][0] + arr_with_index[-2][0] < X:
            continue
            
        left = i+1
        right = n-1
        target = X - a_val
        while left < right:
            b_val, b_idx = arr_with_index[left]
            c_val, c_idx = arr_with_index[right]
            s = b_val + c_val
            if s == target:
                indices = sorted([a_idx, b_idx, c_idx])
                print(f"{indices[0]+1} {indices[1]+1} {indices[2]+1}")
                return
            elif s < target:
                left += 1
            else:
                right -= 1
                
    print(-1)

if __name__ == '__main__':
    main()