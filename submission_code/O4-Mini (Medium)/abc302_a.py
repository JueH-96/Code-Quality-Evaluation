def main():
    import sys
    data = sys.stdin.read().strip().split()
    A = int(data[0])
    B = int(data[1])
    # Minimum number of attacks k such that k*B >= A
    # That's the ceiling of A/B
    result = (A + B - 1) // B
    print(result)

if __name__ == "__main__":
    main()