# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().splitlines()
    N = int(input[0])
    unique_sticks = set()
    for s in input[1:N+1]:
        normalized = min(s, s[::-1])
        unique_sticks.add(normalized)
    print(len(unique_sticks))

if __name__ == "__main__":
    main()