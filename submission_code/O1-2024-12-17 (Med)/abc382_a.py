def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, D = map(int, data[:2])
    S = data[2]

    # Count how many boxes currently contain cookies
    cookies_count = S.count('@')

    # After D cookies are eaten, the number of boxes still containing cookies will be (cookies_count - D).
    # Hence, empty boxes = total boxes - boxes_with_cookies_after = N - (cookies_count - D) = N - cookies_count + D
    answer = N - cookies_count + D

    print(answer)

# Do not forget to call main!
if __name__ == "__main__":
    main()