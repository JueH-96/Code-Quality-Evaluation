def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    
    # Check case 1: subarray in original array sums to S
    def has_subarray_sum(target, arr):
        current_sum = 0
        left = 0
        n = len(arr)
        for right in range(n):
            current_sum += arr[right]
            while current_sum > target and left <= right:
                current_sum -= arr[left]
                left += 1
            if current_sum == target:
                return True
        return False
    
    case1 = has_subarray_sum(S, A)
    if case1:
        print("Yes")
        return
    
    sum_N = sum(A)
    if sum_N > S:
        print("No")
        return
    
    r = S % sum_N
    long_arr = A * 2
    n = len(long_arr)
    current_sum = 0
    left = 0
    found = False
    
    for right in range(n):
        current_sum += long_arr[right]
        while current_sum > S and left <= right:
            current_sum -= long_arr[left]
            left += 1
        if left <= right:
            mod = current_sum % sum_N
            if mod == r:
                found = True
                break
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()