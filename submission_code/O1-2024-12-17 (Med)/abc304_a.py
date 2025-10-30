# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    names = []
    ages = []
    idx = 1
    for _ in range(N):
        S = data[idx]; A = int(data[idx+1])
        idx += 2
        names.append(S)
        ages.append(A)
    
    # Find index of the minimum age
    min_age_index = 0
    min_age_value = ages[0]
    for i in range(1, N):
        if ages[i] < min_age_value:
            min_age_value = ages[i]
            min_age_index = i
    
    # Print names starting from the youngest, in clockwise order
    for i in range(N):
        print(names[(min_age_index + i) % N])

# Do not forget to call main()
main()