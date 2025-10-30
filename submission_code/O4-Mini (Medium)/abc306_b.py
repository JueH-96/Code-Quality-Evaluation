import sys

def main():
    data = sys.stdin.read().split()
    # Expecting exactly 64 tokens, each "0" or "1"
    bits = list(map(int, data))
    result = 0
    for i, b in enumerate(bits):
        if b:
            result |= (1 << i)
    print(result)

if __name__ == "__main__":
    main()