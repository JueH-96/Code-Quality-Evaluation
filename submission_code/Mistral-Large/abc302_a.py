import sys

def calculate_attacks(A, B):
    # Calculate the number of attacks needed
    if A <= 0:
        return 0
    if B <= 0:
        return float('inf')  # Infinite attacks needed if B is non-positive
    return (A + B - 1) // B

def main():
    input = sys.stdin.read()
    A, B = map(int, input.strip().split())
    result = calculate_attacks(A, B)
    print(result)

if __name__ == "__main__":
    main()