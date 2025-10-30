import sys
input = sys.stdin.read

def main():
    data = input().strip()
    A, B = map(int, data.split())
    # Calculate the number of attacks needed
    attacks = (A + B - 1) // B
    print(attacks)

if __name__ == "__main__":
    main()