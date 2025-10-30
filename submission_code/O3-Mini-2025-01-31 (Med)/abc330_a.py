def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # First two numbers are N and L
    N = int(input_data[0])
    L = int(input_data[1])
    # The rest are the scores for the exam
    scores = list(map(int, input_data[2:]))
    passed_count = sum(1 for score in scores if score >= L)
    print(passed_count)

if __name__ == "__main__":
    main()