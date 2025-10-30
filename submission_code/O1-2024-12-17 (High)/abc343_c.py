def main():
    import sys
    N = int(sys.stdin.readline().strip())

    def is_palindrome(num: int) -> bool:
        s = str(num)
        return s == s[::-1]
    
    # We only need to check up to the cube root of N
    import math
    limit = int(N ** (1/3)) + 2
    
    answer = 0
    for x in range(1, limit + 1):
        c = x * x * x
        if c > N:
            break
        if is_palindrome(c):
            answer = c
    
    print(answer)

# Do not forget to call main
if __name__ == "__main__":
    main()