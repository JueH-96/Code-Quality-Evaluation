import sys

def find_good_sequences(L, R):
    def is_power_of_two(n):
        return (n & (n - 1)) == 0

    sequences = []
    current = L

    while current < R:
        # Find the smallest power of two greater than or equal to current
        power_of_two = 1
        while power_of_two <= current:
            power_of_two <<= 1

        # Find the next boundary
        next_boundary = (current + power_of_two - 1) // power_of_two * power_of_two
        if next_boundary > R:
            next_boundary = R

        sequences.append((current, next_boundary))
        current = next_boundary

    return sequences

def main():
    input = sys.stdin.read
    data = input().split()

    L = int(data[0])
    R = int(data[1])

    sequences = find_good_sequences(L, R)

    print(len(sequences))
    for seq in sequences:
        print(seq[0], seq[1])

if __name__ == "__main__":
    main()