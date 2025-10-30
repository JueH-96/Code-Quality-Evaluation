def nand(a, b):
    return int(not (a and b))

def main():
    N = int(input().strip())
    S = input().strip()

    # Convert the input string into a list of integers for faster processing
    A = [int(c) for c in S]

    total_sum = 0
    for i in range(N):
        current_value = A[i]
        for j in range(i, N):
            total_sum += current_value
            if j + 1 < N:
                current_value = nand(current_value, A[j + 1])
            else:
                break

    print(total_sum)

if __name__ == "__main__":
    main()