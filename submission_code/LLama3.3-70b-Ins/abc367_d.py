import sys

def read_ints():
    return map(int, sys.stdin.readline().split())

def main():
    N, M = read_ints()
    A = list(read_ints())

    # Calculate the total distance around the lake
    total_distance = sum(A)

    # Initialize a list to store the cumulative sum of distances
    cumulative_sum = [0] * (N + 1)
    for i in range(N):
        cumulative_sum[i + 1] = cumulative_sum[i] + A[i]

    # Initialize a variable to store the count of possible pairs
    count = 0

    # Iterate over all pairs of rest areas
    for s in range(N):
        for t in range(N):
            if s != t:
                # Calculate the minimum distance between the two rest areas
                distance = cumulative_sum[t + 1] - cumulative_sum[s + 1]
                if distance < 0:
                    distance += total_distance

                # Check if the distance is a multiple of M
                if distance % M == 0:
                    count += 1

    # Print the count of possible pairs
    print(count)

if __name__ == "__main__":
    main()