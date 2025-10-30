def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    def count_trailing_zeros(x):
        count = 0
        while x % 2 == 0:
            x //= 2
            count += 1
        return count
    
    print(count_trailing_zeros(N))

if __name__ == "__main__":
    main()