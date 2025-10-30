# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    S = input[0]
    T = input[1]
    n = len(S)
    m = len(T)

    for w in range(1, n):
        for c in range(1, w + 1):
            result = []
            for i in range(0, n, w):
                substring = S[i:i + w]
                if len(substring) >= c:
                    result.append(substring[c - 1])
            if ''.join(result) == T:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()