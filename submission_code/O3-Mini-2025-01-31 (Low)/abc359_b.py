def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    A = list(map(int, input().split()))
    
    # Initialize dictionary to store the indices of first occurrence
    first_occurrence = {}
    answer = 0
    
    # Loop through the positions, keeping index starting from 0
    for index, color in enumerate(A):
        if color in first_occurrence:
            # When the second occurrence is found, calculate gap between positions
            # Positions in the problem statement are 1-indexed, so comparing indices +1 differences, but difference remains same.
            if index - first_occurrence[color] == 2:
                answer += 1
        else:
            first_occurrence[color] = index

    print(answer)

if __name__ == "__main__":
    main()