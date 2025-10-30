# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    idx = 0

    N = int(data[idx]); idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N

    M = int(data[idx]); idx += 1
    B = list(map(int, data[idx:idx+M]))
    idx += M

    L = int(data[idx]); idx += 1
    C = list(map(int, data[idx:idx+L]))
    idx += L

    Q = int(data[idx]); idx += 1
    X = list(map(int, data[idx:idx+Q]))
    idx += Q

    # Precompute all possible sums of elements from A, B, and C
    sums_abc = set()
    for a in A:
        for b in B:
            for c in C:
                sums_abc.add(a + b + c)

    # Process each query
    answers = []
    for target in X:
        answers.append("Yes" if target in sums_abc else "No")

    print("
".join(answers))

main()