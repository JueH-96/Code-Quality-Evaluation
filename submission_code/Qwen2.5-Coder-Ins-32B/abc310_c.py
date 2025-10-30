# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    sticks = set()
    
    for i in range(1, N + 1):
        s = input[i]
        # Add the string or its reverse, whichever is lexicographically smaller
        sticks.add(min(s, s[::-1]))
    
    print(len(sticks))

if __name__ == "__main__":
    main()