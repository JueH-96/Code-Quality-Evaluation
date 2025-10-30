import sys
input = sys.stdin.read

def find_good_sequences(L, R):
    sequences = []
    current = L
    
    while current < R:
        length = 1
        while current % (2 * length) == 0 and current + 2 * length <= R:
            length *= 2
        sequences.append((current, current + length))
        current += length
    
    return sequences

def main():
    data = input().strip()
    L, R = map(int, data.split())
    
    sequences = find_good_sequences(L, R)
    
    print(len(sequences))
    for l, r in sequences:
        print(l, r)

if __name__ == "__main__":
    main()