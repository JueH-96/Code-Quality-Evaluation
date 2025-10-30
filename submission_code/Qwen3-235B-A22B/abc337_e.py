import sys

def main():
    n = int(sys.stdin.readline())
    m = (n - 1).bit_length()
    friends = [[] for _ in range(m)]
    
    for x in range(1, n + 1):
        x_minus_1 = x - 1
        bit_str = bin(x_minus_1)[2:].zfill(m)
        for i in range(m):
            if bit_str[i] == '1':
                friends[i].append(x)
    
    print(m)
    for bottles in friends:
        print(len(bottles), end='')
        if bottles:
            print(' ' + ' '.join(map(str, bottles)))
        else:
            print()
    
    s = sys.stdin.readline().strip()
    x_prime = int(s, 2) + 1
    print(x_prime)

if __name__ == "__main__":
    main()