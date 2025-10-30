def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    ptr = 1
    
    answers = []
    for _ in range(t):
        n = int(data[ptr]); ptr += 1
        arr = list(map(int, data[ptr:ptr+n]))
        ptr += n

        s = sum(arr)
        # We can only shift values from higher indices to lower indices
        # (i < j => arr[i] += 1, arr[j] -= 1). Hence arr[0] ("A_1") can never
        # be decremented because there's no smaller index than 0.
        #
        # For A to be non-decreasing, we must have
        #   B[0] <= B[1] <= ... <= B[n-1]
        # with B[0] >= arr[0].
        #
        # Consequently, every B[i] >= B[0] >= arr[0]. Thus,
        #   sum(B) = s >= n * arr[0].
        #
        # That is both necessary and sufficient.
        if s >= n * arr[0]:
            answers.append("Yes")
        else:
            answers.append("No")
    
    print("
".join(answers))

# Call main at the end
if __name__ == "__main__":
    main()