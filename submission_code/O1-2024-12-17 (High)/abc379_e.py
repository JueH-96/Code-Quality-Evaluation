def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    dp = 0
    result = 0
    # dp will hold the sum of all substrings that end at the current position
    # result will accumulate the total sum of all substrings
    for i, ch in enumerate(S, start=1):
        digit = int(ch)
        dp = dp * 10 + i * digit
        result += dp
    
    print(result)

# Do not forget to call main
if __name__ == "__main__":
    main()