def main():
    import sys
    data = sys.stdin.read().strip().split()
    A = int(data[0])
    B = int(data[1])
    # Minimum k such that B * k >= A is ceil(A / B)
    result = (A + B - 1) // B
    print(result)

if __name__ == "__main__":
    main()