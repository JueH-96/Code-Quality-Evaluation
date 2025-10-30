# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.readline
    N = int(sys.stdin.readline())
    sticks_set = set()
    for _ in range(N):
        S = sys.stdin.readline().strip()
        canonical = min(S, S[::-1])
        sticks_set.add(canonical)
    print(len(sticks_set))

if __name__ == "__main__":
    main()