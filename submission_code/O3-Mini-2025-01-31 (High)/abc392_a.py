def main():
    import sys
    data = sys.stdin.read().split()
    a, b, c = map(int, data)
    
    # Check if any pairing of two numbers multiplies to the third
    if a * b == c or a * c == b or b * c == a:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()