import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    # Read A and find its maximum
    max_a = -10**18
    for _ in range(n):
        val = int(next(it))
        if val > max_a:
            max_a = val
    # Read B and find its maximum
    max_b = -10**18
    for _ in range(n):
        val = int(next(it))
        if val > max_b:
            max_b = val
    # The answer is the sum of the two maxima
    print(max_a + max_b)

if __name__ == "__main__":
    main()