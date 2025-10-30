import sys

def main():
    S = sys.stdin.readline().rstrip()
    # Replace the last character ('3') with '4'
    print(S[:-1] + '4')

if __name__ == "__main__":
    main()