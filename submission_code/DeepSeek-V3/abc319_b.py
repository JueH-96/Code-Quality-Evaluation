# YOUR CODE HERE
def main():
    N = int(input())
    s = []
    for i in range(N+1):
        min_j = None
        for j in range(1, 10):
            if N % j == 0:
                if i % (N // j) == 0:
                    if min_j is None or j < min_j:
                        min_j = j
        if min_j is not None:
            s.append(str(min_j))
        else:
            s.append('-')
    print(''.join(s))

if __name__ == "__main__":
    main()