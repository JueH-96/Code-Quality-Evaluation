import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    k = int(input[1])
    a = list(map(int, input[2:2+n]))
    
    max_xor = 0
    for combo in itertools.combinations(a, k):
        current_xor = 0
        for num in combo:
            current_xor ^= num
        if current_xor > max_xor:
            max_xor = current_xor
    print(max_xor)

if __name__ == "__main__":
    main()