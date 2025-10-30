def count_set_bits(x):
    return bin(x).count('1')

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    count_even = 0
    count_odd = 0
    
    for a in A:
        if count_set_bits(a) % 2 == 0:
            count_even += 1
        else:
            count_odd += 1
    
    print(max(count_even, count_odd))

if __name__ == "__main__":
    main()