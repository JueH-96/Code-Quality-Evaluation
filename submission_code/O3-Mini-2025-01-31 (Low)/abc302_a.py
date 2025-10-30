def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    A = int(input_data[0])
    B = int(input_data[1])
    # Calculate the minimum number of attacks needed: ceil(A / B)
    attacks = (A + B - 1) // B
    print(attacks)

if __name__ == "__main__":
    main()