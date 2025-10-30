def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    L = int(data[1])
    A = list(map(int, data[2:]))

    # Count how many scores are greater than or equal to L
    passed_count = sum(score >= L for score in A)

    print(passed_count)

# Do NOT forget to call main()
if __name__ == "__main__":
    main()