# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    
    seen = set()
    count = 0
    
    for i in range(N):
        seen.add(S[i])
        count += 1
        if len(seen) == 3:
            print(count)
            return

if __name__ == "__main__":
    main()