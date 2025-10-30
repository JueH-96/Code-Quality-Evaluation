import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Take the last K elements and move them to the front
    result = A[-K:] + A[:-K]
    
    # Print the result as space-separated values
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()