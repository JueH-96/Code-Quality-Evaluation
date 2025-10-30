def main():
    import sys
    data = sys.stdin.read().strip().split()
    # First two integers are N and L
    N = int(data[0])
    L = int(data[1])
    # The next N integers are the scores A_i
    scores = list(map(int, data[2:2+N]))
    # Count how many scores are at least L
    passed_count = sum(1 for s in scores if s >= L)
    print(passed_count)

if __name__ == '__main__':
    main()