# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    S = input().strip()
    result = S.replace("WA", "AC")
    while "WA" in result:
        result = result.replace("WA", "AC")
    print(result)

if __name__ == "__main__":
    main()