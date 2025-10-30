# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    
    MAX_RATING = 500000
    # Difference array for the range effect
    diff = [0] * (MAX_RATING + 2)
    
    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        index += 2
        diff[L] += 1
        if R + 1 <= MAX_RATING:
            diff[R + 1] -= 1
    
    # Calculate the number of contests affecting each rating
    contests_affecting = [0] * (MAX_RATING + 1)
    current_effect = 0
    for i in range(1, MAX_RATING + 1):
        current_effect += diff[i]
        contests_affecting[i] = current_effect
    
    Q = int(data[index])
    index += 1
    
    results = []
    for _ in range(Q):
        X = int(data[index])
        index += 1
        final_rating = X + contests_affecting[X]
        results.append(final_rating)
    
    # Output results
    sys.stdout.write('
'.join(map(str, results)) + '
')