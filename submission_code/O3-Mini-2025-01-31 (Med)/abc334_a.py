def main():
    import sys
    input_data = sys.stdin.read().split()
    B = int(input_data[0])
    G = int(input_data[1])
    if B > G:
        print("Bat")
    else:
        print("Glove")

if __name__ == "__main__":
    main()