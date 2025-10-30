def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read inputs
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))

    # We need to determine the minimal score S for the round N such that the final grade >= X.
    # In each round, final grade = sum(sorted(scores)[1:-1])
    # Try every possible score S from 0 to 100.
    answer = -1
    for s in range(101):
        scores = A + [s]
        scores.sort()
        # Remove the lowest score (first element) and the highest score (last element) then sum rest.
        final_grade = sum(scores[1:-1])
        if final_grade >= X:
            answer = s
            break
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()