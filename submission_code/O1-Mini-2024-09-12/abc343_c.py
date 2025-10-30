def floor_cbrt(n):
    low, high = 0, 10**6
    while low <= high:
        mid = (low + high) // 2
        mid_cubed = mid * mid * mid
        if mid_cubed == n:
            return mid
        elif mid_cubed < n:
            low = mid + 1
        else:
            high = mid - 1
    return high

def is_palindrome(k):
    s = str(k)
    return s == s[::-1]

def main():
    import sys
    N = int(sys.stdin.read())
    x = floor_cbrt(N)
    while x >=1:
        K = x **3
        if is_palindrome(K):
            print(K)
            return
        x -=1
    print(1)

if __name__ == "__main__":
    main()