def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    S = data[2]

    answer = 0
    count = 0  # to count consecutive healthy ('O') teeth

    for char in S:
        if char == 'O':
            count += 1
        else:
            answer += count // K
            count = 0
    # for the last run
    answer += count // K

    print(answer)

if __name__ == "__main__":
    main()