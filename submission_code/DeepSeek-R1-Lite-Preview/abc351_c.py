def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    sequence = []
    for a in A:
        sequence.append(a)
        while len(sequence) >= 2 and sequence[-1] == sequence[-2]:
            x = sequence.pop()
            sequence.pop()
            sequence.append(x + 1)
    print(len(sequence))

if __name__ == "__main__":
    main()