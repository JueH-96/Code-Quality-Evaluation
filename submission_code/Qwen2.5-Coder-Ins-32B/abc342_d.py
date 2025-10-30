# YOUR CODE HERE
import sys
import math
from collections import defaultdict

def is_square(x):
    s = int(math.isqrt(x))
    return s * s == x

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    # Count the frequency of each number
    count = defaultdict(int)
    for a in A:
        count[a] += 1

    # Find the number of valid pairs
    result = 0
    for a in count:
        for b in count:
            if a > b:
                continue
            product = a * b
            if is_square(product):
                if a == b:
                    # If a == b, we need to choose 2 out of count[a] elements
                    result += count[a] * (count[a] - 1) // 2
                else:
                    # If a != b, we need to choose one from count[a] and one from count[b]
                    result += count[a] * count[b]

    print(result)

if __name__ == "__main__":
    main()