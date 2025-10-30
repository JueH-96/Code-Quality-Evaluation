# YOUR CODE HERE
import sys

def main():
    S = sys.stdin.read().strip()
    
    if S[:3] == "ABC" and S[3:].isdigit():
        num = int(S[3:])
        if (1 <= num <= 315) or (317 <= num <= 349):
            print("Yes")
        else:
            print("No")
    else:
        print("No")

if __name__ == "__main__":
    main()