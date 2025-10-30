# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip()
    S, T = data.split()
    if S == "AtCoder" and T == "Land":
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()