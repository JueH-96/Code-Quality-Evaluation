# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().strip()
    S = list(map(int, input.split()))
    
    # Check if the sequence is monotonically non-decreasing
    for i in range(1, 8):
        if S[i] < S[i-1]:
            print("No")
            return
    
    # Check if all elements are between 100 and 675, inclusive
    for s in S:
        if s < 100 or s > 675:
            print("No")
            return
    
    # Check if all elements are multiples of 25
    for s in S:
        if s % 25 != 0:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()