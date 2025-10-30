import sys
from fractions import Fraction

def expected_cost(N, M, wheels):
    # Initialize the expected cost array
    expected_cost = [Fraction(0)] * (M + 1)
    expected_cost[0] = Fraction(0)

    for i in range(1, M + 1):
        min_cost = float('inf')
        for wheel in wheels:
            cost, probs = wheel
            expected = Fraction(cost)
            for prob in probs:
                if i - prob >= 0:
                    expected += probs[prob] * expected_cost[i - prob]
            min_cost = min(min_cost, expected)
        expected_cost[i] = min_cost

    return float(expected_cost[M])

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    wheels = []
    for _ in range(N):
        C = int(data[index])
        P = int(data[index + 1])
        index += 2
        S = [int(data[index + j]) for j in range(P)]
        index += P

        probs = {}
        for s in S:
            if s in probs:
                probs[s] += Fraction(1, P)
            else:
                probs[s] = Fraction(1, P)

        wheels.append((C, probs))

    result = expected_cost(N, M, wheels)
    print(result)

if __name__ == "__main__":
    main()