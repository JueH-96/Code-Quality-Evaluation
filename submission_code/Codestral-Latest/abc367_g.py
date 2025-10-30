MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:]))

    # Calculate the total XOR of the entire sequence A
    total_xor = 0
    for num in A:
        total_xor ^= num

    # Calculate the result based on the given formula
    if N % M == 0:
        result = pow(total_xor, K, MOD)
    else:
        result = 0

    print(result)

if __name__ == "__main__":
    main()