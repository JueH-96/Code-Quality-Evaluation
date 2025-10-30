# YOUR CODE HERE
import sys

def main():
    import sys
    import math

    N = int(sys.stdin.readline().strip())
    M = math.ceil(math.log2(N))
    print(M)
    sys.stdout.flush()

    friends = []
    for i in range(M):
        start = 1 << i
        bottles = []
        for j in range(N):
            if (j + 1) & start:
                bottles.append(j + 1)
        print(len(bottles), *bottles)
        sys.stdout.flush()
        friends.append(bottles)

    S = sys.stdin.readline().strip()
    spoiled = 0
    for i in range(M):
        if S[i] == '1':
            spoiled |= (1 << i)

    print(spoiled + 1)
    sys.stdout.flush()

if __name__ == "__main__":
    main()