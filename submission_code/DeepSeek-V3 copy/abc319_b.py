# YOUR CODE HERE
def main():
    N = int(input())
    divisors = []
    for j in range(1, 10):
        if N % j == 0:
            divisors.append(j)
    s = []
    for i in range(N + 1):
        candidates = []
        for j in divisors:
            if i % (N // j) == 0:
                candidates.append(j)
        if candidates:
            s.append(str(min(candidates)))
        else:
            s.append('-')
    print(''.join(s))

if __name__ == "__main__":
    main()