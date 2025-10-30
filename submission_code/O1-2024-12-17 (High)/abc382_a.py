def main():
    import sys

    data = sys.stdin.read().split()
    N, D = map(int, data[:2])
    S = data[2]

    # Count how many boxes initially contain a cookie
    cookie_count = S.count('@')
    
    # After D days, D cookies will be eaten, each from a distinct box.
    # The number of empty boxes is:
    # (number of boxes that started empty) + (number of boxes from which a cookie was eaten)
    # Which simplifies to: N - cookie_count + D
    answer = (N - cookie_count) + D
    print(answer)

# Do not forget to call the main function
if __name__ == "__main__":
    main()