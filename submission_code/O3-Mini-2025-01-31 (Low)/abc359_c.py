def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    Sx = int(data[0])
    Sy = int(data[1])
    Tx = int(data[2])
    Ty = int(data[3])
    # The minimal toll equals the vertical difference.
    result = abs(Ty - Sy)
    sys.stdout.write(str(result))
    
if __name__ == "__main__":
    main()