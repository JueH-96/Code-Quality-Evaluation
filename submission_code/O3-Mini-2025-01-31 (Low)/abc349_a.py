def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))
    total_known = sum(A)
    # The sum of final scores must be 0, so the final score for person N is:
    final_score_N = -total_known
    print(final_score_N)

if __name__ == "__main__":
    main()