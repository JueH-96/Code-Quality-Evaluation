def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # The first input value is the integer N (total number of persons)
    N = int(input_data[0])

    # The next N-1 values are the final scores for persons 1 through N-1
    scores = list(map(int, input_data[1:]))

    # Since each game increases one person's score by +1 and reduces another's by -1,
    # the sum of all changes is 0. With everyone starting from 0, the total final score must be 0.
    # Hence, the final score of person N is just the negative sum of the given scores.
    person_N_score = -sum(scores)

    # Output the final score for person N.
    print(person_N_score)

if __name__ == '__main__':
    main()