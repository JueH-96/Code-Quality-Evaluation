# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    photos = []
    index = 2
    for _ in range(M):
        photos.append(list(map(int, input[index:index + N])))
        index += N

    # Create a set to store pairs that stood next to each other
    next_to_each_other = set()
    for photo in photos:
        for i in range(N - 1):
            a, b = photo[i], photo[i + 1]
            if a > b:
                a, b = b, a
            next_to_each_other.add((a, b))

    # Calculate the total number of pairs
    total_pairs = N * (N - 1) // 2

    # Subtract the number of pairs that stood next to each other
    bad_mood_pairs = total_pairs - len(next_to_each_other)

    print(bad_mood_pairs)

if __name__ == "__main__":
    main()