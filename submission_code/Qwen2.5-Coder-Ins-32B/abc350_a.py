# YOUR CODE HERE
import sys

def main():
    S = sys.stdin.read().strip()
    if S.startswith("ABC"):
        num = int(S[3:])
        if (1 <= num <= 314) or (315 <= num <= 348) or (num == 349):
            print("Yes")
        else:
            print("No")
    else:
        print("No")

if __name__ == "__main__":
    main()