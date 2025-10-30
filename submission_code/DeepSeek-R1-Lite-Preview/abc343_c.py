def integer_cube_root(N):
    low = 0
    high = N
    while low <= high:
        mid = (low + high) // 2
        cube = mid ** 3
        if cube == N:
            return mid
        elif cube < N:
            low = mid + 1
        else:
            high = mid - 1
    return high

def is_palindrome(K):
    s = str(K)
    return s == s[::-1]

def main():
    N = int(input().strip())
    x_max = integer_cube_root(N)
    for x in range(x_max, 0, -1):
        K = x ** 3
        if is_palindrome(K):
            print(K)
            return
    print(0)

if __name__ == "__main__":
    main()