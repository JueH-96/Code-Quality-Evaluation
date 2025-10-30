# YOUR CODE HERE
import sys

def is_subsequence(s, t):
    it = iter(s)
    return all(c in it for c in t)

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    T = input[1]
    S = input[2:]

    count = 0
    for i in range(N):
        for j in range(N):
            if is_subsequence(S[i] + S[j], T):
                count += 1

    print(count)

if __name__ == "__main__":
    main()