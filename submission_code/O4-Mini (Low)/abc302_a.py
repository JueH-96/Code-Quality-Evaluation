import sys

def main():
    data = sys.stdin.read().strip().split()
    A = int(data[0])
    B = int(data[1])
    # We need the smallest x such that x*B >= A
    # That is ceil(A/B) = (A + B - 1) // B
    result = (A + B - 1) // B
    print(result)

if __name__ == "__main__":
    main()