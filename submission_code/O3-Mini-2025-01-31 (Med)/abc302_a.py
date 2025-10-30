def main():
    import sys
    input_line = sys.stdin.read().strip()
    if not input_line:
        return
    parts = input_line.split()
    A = int(parts[0])
    B = int(parts[1])
    
    # Compute the minimum number of attacks required:
    # We need k * B >= A, so k is the ceiling of (A / B)
    # Using integer arithmetic to compute the ceiling:
    k = (A + B - 1) // B
    print(k)

if __name__ == '__main__':
    main()