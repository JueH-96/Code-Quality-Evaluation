def main():
    import sys
    data = sys.stdin.read().split()
    N, D = map(int, data[:2])
    S = data[2]
    total_cookies = S.count('@')
    # After eating D cookies, remaining cookies = total_cookies - D
    # Empty boxes = N - remaining_cookies
    result = N - (total_cookies - D)
    print(result)

if __name__ == "__main__":
    main()