import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    
    # Maximum possible rating value
    MAX_RATING = 500000
    
    # We will use a difference array to manage the increments efficiently
    increments = [0] * (MAX_RATING + 2)  # +2 to handle boundary safely
    
    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        index += 2
        increments[L] += 1
        increments[R + 1] -= 1
    
    # Build the actual increments array using prefix sum
    for i in range(1, MAX_RATING + 2):
        increments[i] += increments[i - 1]
    
    # Now, increments[i] tells how much the rating i should be incremented after all contests
    
    Q = int(data[index])
    index += 1
    
    results = []
    for _ in range(Q):
        X = int(data[index])
        index += 1
        # The final rating after all contests
        final_rating = X + increments[X]
        results.append(str(final_rating))
    
    # Output all results
    print("
".join(results))

if __name__ == "__main__":
    main()