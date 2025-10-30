import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    k = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    max_xor = 0
    for combo in itertools.combinations(a, k):
        current = 0
        for num in combo:
            current ^= num
        if current > max_xor:
            max_xor = current
    print(max_xor)

if __name__ == "__main__":
    main()