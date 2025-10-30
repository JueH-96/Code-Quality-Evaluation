# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    result = str(N) * N
    print(result)

if __name__ == "__main__":
    main()