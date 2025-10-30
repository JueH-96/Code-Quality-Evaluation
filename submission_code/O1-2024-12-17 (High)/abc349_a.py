def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    scores = list(map(int, data[1:]))

    # Since the sum of all final scores (including person N) remains 0,
    # person N's score is simply the negative of the sum of the given scores.
    ans = -sum(scores)
    print(ans)

# Do not forget to call the main function
main()