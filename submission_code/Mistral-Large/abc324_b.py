import sys

def is_power_of_two_and_three(N):
    # Check if N can be expressed as 2^x * 3^y
    original_N = N
    while N % 2 == 0:
        N //= 2
    while N % 3 == 0:
        N //= 3
    return N == 1

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    if is_power_of_two_and_three(N):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()