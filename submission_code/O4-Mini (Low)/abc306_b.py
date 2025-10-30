import sys

def main():
    data = sys.stdin.read().strip().split()
    # Expect 64 values
    if not data:
        return
    bits = list(map(int, data))
    # Compute the integer value: sum of bits[i] * 2**i
    result = 0
    for i, b in enumerate(bits):
        if b:
            result |= (1 << i)
    print(result)

if __name__ == "__main__":
    main()