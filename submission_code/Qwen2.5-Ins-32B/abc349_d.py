import math

def find_good_sequences(L, R):
    sequences = []
    while L < R:
        # Find the largest power of 2 that divides the difference
        diff = R - L
        power_of_2 = int(math.log2(diff))
        # Adjust the range to fit the largest possible good sequence
        if diff & (1 << power_of_2) == 0:
            power_of_2 -= 1
        r = L + (1 << power_of_2)
        sequences.append((L, r))
        L = r
    return sequences

def main():
    L, R = map(int, input().split())
    sequences = find_good_sequences(L, R)
    print(len(sequences))
    for l, r in sequences:
        print(l, r)

if __name__ == "__main__":
    main()