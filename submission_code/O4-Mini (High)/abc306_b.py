import sys

def main():
    data = sys.stdin.read().split()
    # data should contain 64 entries A_0 through A_63
    ans = 0
    for i, bit in enumerate(data):
        if bit == '1':
            ans |= 1 << i
    print(ans)

if __name__ == "__main__":
    main()