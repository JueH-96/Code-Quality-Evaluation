import sys

def main():
    # Read all tokens (they might be split across several lines)
    tokens = sys.stdin.read().strip().split()
    
    # We expect exactly 64 bits, but in any case take first 64 after converting to int
    bits = list(map(int, tokens[:64]))
    
    result = 0
    for i, bit in enumerate(bits):
        if bit:          # same as "if bit == 1"
            result |= 1 << i   # add 2^i
    
    print(result)

if __name__ == "__main__":
    main()